"""Chain four NVFP4 projections and the BF16 VSA boundary in one Hazy launch."""

import json
import os
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiler import ProjectionTaskParams, compile_projection_attention, compile_projection_sparse
from attention_boundary import read_tensor
from projection_parity import CHECKPOINT, REVISION, compare

import torch
from flashinfer import SfLayout
from huggingface_hub import snapshot_download
from safetensors import safe_open

from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize
from mk_h3_projection import prepare, params_words, run_attention, run_projection_sparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse", action="store_true",
                        help="chain the exact-mask BF16 fine attention in the same Hazy launch")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("RTX PRO 6000 SM120 required")
    root = Path(os.environ["H3MK_ATTENTION_CAPTURE"])
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("verified") or manifest["capture_mode"] != "attention_inputs":
        raise RuntimeError("missing verified first-block attention capture")
    if manifest["q_norm_eps"] != manifest["k_norm_eps"]:
        raise RuntimeError("Q/K normalization epsilons differ")
    names = ("to_q", "to_k", "to_v", "to_gate_compress")
    source = read_tensor(root, manifest, "to_q_input")
    expected = [read_tensor(root, manifest, f"{name}_output") for name in names]
    m, k, n = 37771, 5376, 7168
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if sm_count != 188:
        raise RuntimeError(f"unexpected SM count: {sm_count}")
    params = torch.empty(params_words() * 4, device="cuda", dtype=torch.int32)
    packed_refs = []
    projection_outputs = []
    targets = []
    checkpoint = snapshot_download(CHECKPOINT, revision=REVISION, local_files_only=True)
    with safe_open(f"{checkpoint}/transformer/nvfp4_weights.safetensors", framework="pt", device="cpu") as reader:
        for index, name in enumerate(names):
            prefix = f"transformer_blocks.0.attn.{name}"
            weight = reader.get_tensor(f"{prefix}::_nvfp4_weight").cuda()
            weight_sf = reader.get_tensor(f"{prefix}::_nvfp4_weight_scale").cuda()
            alpha_raw = reader.get_tensor(f"{prefix}::_nvfp4_alpha").cuda()
            sf_key = f"{prefix}::_nvfp4_input_global_sf"
            input_sf = (reader.get_tensor(sf_key).cuda() if sf_key in reader.keys()
                        else torch.tensor(1.0, device="cuda", dtype=torch.float32))
            packed, packed_sf = _nvfp4_quantize(source.reshape(m, k).cuda(), input_sf,
                                                 sfLayout=SfLayout.layout_128x4,
                                                 do_shuffle=False)
            alpha = float((alpha_raw.float() / input_sf.float()).item())
            output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            prepare(params.data_ptr() + index * params_words() * 4,
                    packed.data_ptr(), weight.data_ptr(), packed_sf.data_ptr(),
                    weight_sf.data_ptr(), output.data_ptr(), m, n, k,
                    alpha, sm_count)
            packed_refs.append((packed, packed_sf, weight, weight_sf))
            projection_outputs.append(output)
            targets.append(expected[index].reshape(m, n).cuda())

    q_weight = read_tensor(root, manifest, "q_norm_weight").cuda()
    k_weight = read_tensor(root, manifest, "k_norm_weight").cuda()
    cos = read_tensor(root, manifest, "rope_cos").cuda()
    sin = read_tensor(root, manifest, "rope_sin").cuda()
    index = read_tensor(root, manifest, "untile_combined_index").cuda()
    reference_root = Path(os.environ.get(
        "H3MK_REFERENCE_CAPTURE", "/cache/runs/reference-bf16-vsa-20261115"))
    reference = json.loads((reference_root / "reference_manifest.json").read_text())
    tiled = {}
    for name in ("q", "k", "v", "gate"):
        oracle = read_tensor(reference_root, reference, f"step0_block0_{name}").cuda()
        tiled[name] = (torch.zeros_like(oracle), oracle)
    if args.sparse:
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn

        mask = read_tensor(reference_root, reference, "step0_block0_vsa_mask").cuda()
        valid = read_tensor(reference_root, reference, "step0_block0_valid_lengths").cuda()
        frozen_q, frozen_k, frozen_v = (tiled[name][1] for name in ("q", "k", "v"))
        expected_bhsd, _ = block_sparse_attn(
            frozen_q.transpose(1, 2).contiguous(),
            frozen_k.transpose(1, 2).contiguous(),
            frozen_v.transpose(1, 2).contiguous(), mask, valid)
        expected_fine = expected_bhsd.transpose(1, 2).contiguous()
        sparse_output = torch.empty_like(expected_fine)

    slots = 1024
    probe = torch.zeros((slots, 1, 32), device="cuda", dtype=torch.int32)
    probe[:, 0, 0] = 1
    timings = torch.zeros((slots, 10 if args.sparse else 8, 128), device="cuda", dtype=torch.int32)
    counts = torch.zeros(sm_count, device="cuda", dtype=torch.int32)
    ids = torch.full((sm_count,), -1, device="cuda", dtype=torch.int32)
    rendezvous = torch.zeros(1, device="cuda", dtype=torch.int32)
    dummy_float = torch.zeros(1, device="cuda", dtype=torch.float32)
    wave_counts = torch.zeros(2 if args.sparse else 1, device="cuda", dtype=torch.int32)
    graph_error = torch.zeros(1, device="cuda", dtype=torch.int32)

    def launch(instructions):
        common = (instructions, timings, counts, ids, rendezvous,
                  dummy_float, wave_counts, graph_error, params,
                  projection_outputs[0].view(-1), projection_outputs[1].view(-1),
                  tiled["q"][0].view(-1), tiled["k"][0].view(-1),
                  q_weight.view(-1), k_weight.view(-1), cos.view(-1), sin.view(-1),
                  projection_outputs[2].view(-1), projection_outputs[3].view(-1),
                  tiled["v"][0].view(-1), tiled["gate"][0].view(-1),
                  index.view(-1), sm_count)
        if args.sparse:
            run_projection_sparse(*common, mask.view(-1), valid.view(-1),
                                  sparse_output.view(-1), stream=torch.cuda.current_stream())
        else:
            run_attention(*common, stream=torch.cuda.current_stream())

    launch(probe)
    torch.cuda.synchronize()
    measured = ids.cpu().tolist()
    if len(set(measured)) != sm_count or min(measured) < 0 or max(measured) >= slots:
        raise RuntimeError("physical SM discovery failed")
    tasks = tuple(ProjectionTaskParams(m, n, k, sm_count, i) for i in range(4))
    compile_stream = compile_projection_sparse if args.sparse else compile_projection_attention
    instructions = torch.tensor(
        compile_stream(tasks, m, manifest["q_norm_eps"], measured, slots),
        device="cuda", dtype=torch.int32)
    for rep in range(2 if args.sparse else 3):
        for output in projection_outputs:
            output.fill_(float("nan"))
        for output, _ in tiled.values():
            output.zero_()
        if args.sparse:
            sparse_output.fill_(float("nan"))
        counts.zero_()
        ids.fill_(-1)
        wave_counts.zero_()
        graph_error.zero_()
        launch(instructions)
        torch.cuda.synchronize()
        if not torch.equal(wave_counts, torch.full_like(wave_counts, sm_count)) or int(graph_error[0]):
            raise RuntimeError("cross-SM projection-to-attention barrier failed")
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise RuntimeError("Hazy chained instruction stream did not complete")
        for name, actual, expected_output in zip(names, projection_outputs, targets, strict=True):
            compare(f"chained Hazy {name} rep={rep}", actual, expected_output)
        for name, (actual, oracle) in tiled.items():
            diff = (actual.float() - oracle.float()).abs()
            mismatch = int((actual != oracle).count_nonzero())
            outside = int((~torch.isclose(actual.float(), oracle.float(),
                                          rtol=0.015625, atol=0.03125)).count_nonzero())
            print(f"chained Hazy tiled {name} rep={rep}: max_abs_error={float(diff.max())} "
                  f"mismatched_elements={mismatch} outside_tolerance={outside}", flush=True)
            if outside or (name in ("v", "gate") and mismatch):
                raise AssertionError(f"chained {name}: tiled output differs from reference")
        if args.sparse:
            diff = (sparse_output.float() - expected_fine.float()).abs()
            outside = int((~torch.isclose(sparse_output.float(), expected_fine.float(),
                                          rtol=0.015625, atol=0.03125)).count_nonzero())
            print(f"chained Hazy sparse attention rep={rep}: max_abs_error={float(diff.max())} "
                  f"mean_abs_error={float(diff.mean())} outside_tolerance={outside} "
                  f"elements={diff.numel()}", flush=True)
            if outside:
                raise AssertionError("chained sparse attention exceeds BF16 tolerance")
    print("PASS: NVFP4 Q/K/V/gate projections and BF16 VSA boundary in one Hazy launch", flush=True)
    if args.sparse:
        print("PASS: exact-mask BF16 sparse fine attention in the same Hazy launch", flush=True)


if __name__ == "__main__":
    main()
