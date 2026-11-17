"""Run first-block BF16 Q/K norm and partial RoPE inside Hazy on SM120."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiler import compile_qknorm_rope
from attention_boundary import read_tensor

import torch
from mk_h3_projection import run_attention


def main() -> None:
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("RTX PRO 6000 SM120 required")
    root = Path(os.environ["H3MK_ATTENTION_CAPTURE"])
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("verified") or manifest["capture_mode"] != "attention_inputs":
        raise RuntimeError("missing verified attention capture")
    q = read_tensor(root, manifest, "to_q_output").contiguous().cuda()
    k = read_tensor(root, manifest, "to_k_output").contiguous().cuda()
    expected_q = read_tensor(root, manifest, "q_pre_tile").contiguous().cuda()
    expected_k = read_tensor(root, manifest, "k_pre_tile").contiguous().cuda()
    q_out = torch.empty_like(expected_q)
    k_out = torch.empty_like(expected_k)
    q_weight = read_tensor(root, manifest, "q_norm_weight").cuda()
    k_weight = read_tensor(root, manifest, "k_norm_weight").cuda()
    cos = read_tensor(root, manifest, "rope_cos").cuda()
    sin = read_tensor(root, manifest, "rope_sin").cuda()
    v = read_tensor(root, manifest, "to_v_output").contiguous().cuda()
    gate = read_tensor(root, manifest, "to_gate_compress_output").contiguous().cuda()
    index = read_tensor(root, manifest, "untile_combined_index").cuda()
    if q.shape != (1, 37771, 7168) or expected_q.shape != (1, 37771, 56, 128):
        raise RuntimeError("unexpected first-block Q geometry")
    if manifest["q_norm_eps"] != manifest["k_norm_eps"]:
        raise RuntimeError("Q/K normalization epsilons differ")
    eps = manifest["q_norm_eps"]
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if sm_count != 188:
        raise RuntimeError(f"unexpected SM count: {sm_count}")
    slots = 1024
    probe = torch.zeros((slots, 1, 32), device="cuda", dtype=torch.int32)
    probe[:, 0, 0] = 1
    timings = torch.zeros((slots, 3, 128), device="cuda", dtype=torch.int32)
    counts = torch.zeros(sm_count, device="cuda", dtype=torch.int32)
    ids = torch.full((sm_count,), -1, device="cuda", dtype=torch.int32)
    rendezvous = torch.zeros(1, device="cuda", dtype=torch.int32)
    dummy_float = torch.zeros(1, device="cuda", dtype=torch.float32)
    dummy_int = torch.zeros(1, device="cuda", dtype=torch.int32)

    def launch(instructions, q_target, k_target, v_target, gate_target):
        run_attention(instructions, timings, counts, ids, rendezvous,
                      dummy_float, dummy_int, dummy_int, dummy_int,
                      q.view(-1), k.view(-1), q_target.view(-1), k_target.view(-1),
                      q_weight.view(-1), k_weight.view(-1), cos.view(-1), sin.view(-1),
                      v.view(-1), gate.view(-1), v_target.view(-1), gate_target.view(-1),
                      index.view(-1),
                      sm_count, stream=torch.cuda.current_stream())

    launch(probe, q_out, k_out, dummy_int, dummy_int)
    torch.cuda.synchronize()
    measured = ids.cpu().tolist()
    if len(set(measured)) != sm_count or min(measured) < 0 or max(measured) >= slots:
        raise RuntimeError("physical SM discovery failed")
    instructions = torch.tensor(compile_qknorm_rope(37771, eps, measured, slots),
                                device="cuda", dtype=torch.int32)
    results = {}
    for rep in range(3):
        q_out.fill_(float("nan"))
        k_out.fill_(float("nan"))
        counts.zero_()
        ids.fill_(-1)
        launch(instructions, q_out, k_out, dummy_int, dummy_int)
        torch.cuda.synchronize()
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise RuntimeError("Hazy Q/K norm stream did not complete")
        for name, actual, expected in (("q", q_out, expected_q), ("k", k_out, expected_k)):
            diff = (actual.float() - expected.float()).abs()
            close = torch.isclose(actual.float(), expected.float(), rtol=0.015625, atol=0.03125)
            result = {"max_abs_error": float(diff.max()),
                      "mean_abs_error": float(diff.mean()),
                      "mismatched_elements": int((actual != expected).count_nonzero()),
                      "outside_tolerance": int((~close).count_nonzero()),
                      "elements": actual.numel()}
            print(f"Hazy {name} norm/RoPE rep={rep}: {json.dumps(result)}", flush=True)
            if result["outside_tolerance"]:
                raise AssertionError(f"{name}: Q/K norm exceeds BF16 tolerance")
            results[name] = result
    reference_root = Path(os.environ.get(
        "H3MK_REFERENCE_CAPTURE", "/cache/runs/reference-bf16-vsa-20261115"))
    reference = json.loads((reference_root / "reference_manifest.json").read_text())
    tiled_outputs = {}
    for name in ("q", "k", "v", "gate"):
        expected = read_tensor(reference_root, reference, f"step0_block0_{name}").cuda()
        tiled_outputs[name] = (torch.zeros_like(expected), expected)
    tiled_instructions = torch.tensor(
        compile_qknorm_rope(37771, eps, measured, slots, tile=True),
        device="cuda", dtype=torch.int32)
    for rep in range(3):
        for actual, _ in tiled_outputs.values():
            actual.zero_()
        counts.zero_()
        ids.fill_(-1)
        launch(tiled_instructions, tiled_outputs["q"][0], tiled_outputs["k"][0],
               tiled_outputs["v"][0], tiled_outputs["gate"][0])
        torch.cuda.synchronize()
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise RuntimeError("Hazy Q/K norm + tile stream did not complete")
        for name, (actual, expected) in tiled_outputs.items():
            delta = (actual.float() - expected.float()).abs()
            outside = int((~torch.isclose(actual.float(), expected.float(),
                                          rtol=0.015625, atol=0.03125)).count_nonzero())
            mismatch = int((actual != expected).count_nonzero())
            print(f"Hazy tiled {name} rep={rep}: max_abs_error={float(delta.max())} "
                  f"mismatched_elements={mismatch} outside_tolerance={outside}", flush=True)
            if outside or (name in ("v", "gate") and mismatch):
                raise AssertionError(f"{name}: tiled output differs from frozen VSA input")
    print(json.dumps({"verified": True, "operation": "Hazy Q/K norm + partial RoPE",
                      "shape": list(expected_q.shape), "repeats": 3,
                      "tile_count": 668, "tiled_against_frozen_reference": True,
                      "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
