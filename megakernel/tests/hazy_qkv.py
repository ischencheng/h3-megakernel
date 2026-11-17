"""Step-0 block-0 Q/K/V and active gate in one Hazy instruction stream."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiler import ProjectionTaskParams, compile_projections

import torch
from flashinfer import SfLayout
from huggingface_hub import snapshot_download
from safetensors import safe_open

from fastvideo.layers.quantization.nvfp4_config import _mm_fp4, _nvfp4_quantize
from mk_h3_projection import prepare, params_words, resources, run
from projection_parity import CHECKPOINT, REVISION, captured_tensors, compare


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("RTX PRO 6000 SM120 required")
    reference = Path(os.environ["H3MK_PROJECTION_CAPTURE"])
    manifest = json.loads((reference / "manifest.json").read_text())
    names = ["to_q", "to_k", "to_v"]
    if manifest.get("gate_active") is True:
        names.append("to_gate_compress")
    labels = tuple(f"{name}_output" for name in names[1:])
    captured = captured_tensors(labels)
    source, expected = captured[0], captured[1:]
    m, k, n = 37771, 5376, 7168
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if sm_count != 188:
        raise RuntimeError(f"unexpected physical SM count: {sm_count}")
    slots = 1024
    packed_inputs, packed_scales, outputs, expected_cuda = [], [], [], []
    params = torch.empty(params_words() * len(names), device="cuda", dtype=torch.int32)
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
            x = source.reshape(-1, k).contiguous().cuda()
            packed, packed_sf = _nvfp4_quantize(x, input_sf,
                                                 sfLayout=SfLayout.layout_128x4,
                                                 do_shuffle=False)
            alpha = (alpha_raw.float() / input_sf.float()).reshape(())
            target = expected[index].reshape(m, n).contiguous().cuda()
            reconstructed = _mm_fp4(packed, weight.T, packed_sf, weight_sf.T,
                                    alpha, torch.bfloat16, None, backend="auto")
            compare(f"FastVideo {name} vs capture", reconstructed, target)
            output = torch.empty_like(target)
            prepare(params.data_ptr() + index * params_words() * 4,
                    packed.data_ptr(), weight.data_ptr(), packed_sf.data_ptr(),
                    weight_sf.data_ptr(), output.data_ptr(), m, n, k,
                    float(alpha.item()), sm_count)
            packed_inputs.append((packed, weight))
            packed_scales.append((packed_sf, weight_sf))
            outputs.append(output)
            expected_cuda.append(target)
    metadata = dict(resources(m, n, k))
    if metadata["threads_per_cta"] != 384 or metadata["shared_bytes"] > 88480:
        raise RuntimeError("Hazy projection resource contract changed")
    print("QKV/Gate resources", json.dumps(metadata), "projections", names, flush=True)

    probe = torch.zeros((slots, 1, 32), device="cuda", dtype=torch.int32)
    probe[:, 0, 0] = 1
    probe_times = torch.zeros((slots, 1, 128), device="cuda", dtype=torch.int32)
    counts = torch.zeros(sm_count, device="cuda", dtype=torch.int32)
    ids = torch.full((sm_count,), -1, device="cuda", dtype=torch.int32)
    rendezvous = torch.zeros(1, device="cuda", dtype=torch.int32)
    dummy_float = torch.zeros(1, device="cuda", dtype=torch.float32)
    dummy_int = torch.zeros(1, device="cuda", dtype=torch.int32)
    run(probe, probe_times, counts, ids, rendezvous, dummy_float,
        dummy_int, dummy_int, params, sm_count,
        stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    measured_ids = ids.cpu().tolist()
    if len(set(measured_ids)) != sm_count or min(measured_ids) < 0 or max(measured_ids) >= slots:
        raise RuntimeError("physical SM discovery failed")
    tasks = tuple(ProjectionTaskParams(m, n, k, sm_count, index)
                  for index in range(len(names)))
    instructions = torch.tensor(compile_projections(tasks, measured_ids, slots),
                                device="cuda", dtype=torch.int32)
    timings = torch.zeros((slots, len(names) + 2, 128), device="cuda", dtype=torch.int32)
    for rep in range(3):
        for output in outputs:
            output.fill_(float("nan"))
        counts.zero_()
        ids.fill_(-1)
        run(instructions, timings, counts, ids, rendezvous, dummy_float,
            dummy_int, dummy_int, params, sm_count,
            stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise RuntimeError("ordinary Hazy instructions did not complete")
        for name, output, target in zip(names, outputs, expected_cuda, strict=True):
            compare(f"Hazy {name} vs capture rep={rep}", output, target)
    print("PASS: Q/K/V and active gate in one Hazy instruction stream", flush=True)


if __name__ == "__main__":
    main()
