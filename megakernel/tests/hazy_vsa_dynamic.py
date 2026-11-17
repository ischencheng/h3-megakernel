"""Validate one-launch dynamic mask construction and BF16 VSA attention."""

import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_boundary import read_tensor
from compiler import compile_vsa_dynamic, compile_vsa_full

import torch
from fastvideo_kernel.block_sparse_attn import block_sparse_attn
from mk_h3_projection import run_vsa_dynamic, run_vsa_full


def main() -> None:
    full = "--compress" in sys.argv[1:]
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 required")
    root = Path(os.environ["H3MK_REFERENCE_CAPTURE"])
    manifest = json.loads((root / "reference_manifest.json").read_text())
    if not manifest.get("verified") or manifest["env"]["FASTVIDEO_H3_VSA_FP4"] != "0":
        raise RuntimeError("verified BF16 VSA capture required")
    policy = manifest["step0_block0_vsa_policy"]
    if not policy["exempt"] or len(policy["video_tile_spans"]) != 1:
        raise RuntimeError("dynamic gate requires one exempt video region")
    start_tile, end_tile = policy["video_tile_spans"][0]
    keep = math.ceil((1 - policy["span_sparsities"][0]) * (end_tile - start_tile))
    q, k, v = (read_tensor(root, manifest, f"step0_block0_{name}").cuda()
               for name in ("q", "k", "v"))
    valid = read_tensor(root, manifest, "step0_block0_valid_lengths").cuda()
    expected_mask = read_tensor(root, manifest, "step0_block0_vsa_mask").cuda()
    if full:
        expected = read_tensor(root, manifest, "step0_block0_attention").cuda()
        gate = read_tensor(root, manifest, "step0_block0_gate").cuda()
    else:
        expected_bhsd, _ = block_sparse_attn(
            q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(), expected_mask, valid)
        expected = expected_bhsd.transpose(1, 2).contiguous()
        gate = torch.zeros_like(q)
    q_pool = torch.empty((1, 56, 668, 128), dtype=torch.float32, device="cuda")
    k_pool = torch.empty_like(q_pool)
    v_pool = torch.empty_like(q_pool)
    scores = torch.empty((1, 56, 668, 668), dtype=torch.float32, device="cuda")
    mask = torch.empty_like(expected_mask)
    output = torch.empty_like(q)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if sm_count != 188:
        raise RuntimeError(f"unexpected SM count: {sm_count}")
    slots = 1024
    probe = torch.zeros((slots, 1, 32), device="cuda", dtype=torch.int32)
    probe[:, 0, 0] = 1
    timings = torch.zeros((slots, 12 if full else 9, 128), device="cuda", dtype=torch.int32)
    counts = torch.zeros(sm_count, device="cuda", dtype=torch.int32)
    ids = torch.full((sm_count,), -1, device="cuda", dtype=torch.int32)
    rendezvous = torch.zeros(1, device="cuda", dtype=torch.int32)
    dummy_float = torch.zeros(1, device="cuda", dtype=torch.float32)
    wave_counts = torch.zeros(4 if full else 3, device="cuda", dtype=torch.int32)
    graph_error = torch.zeros(1, device="cuda", dtype=torch.int32)

    def launch(instructions):
        kernel = run_vsa_full if full else run_vsa_dynamic
        kernel(instructions, timings, counts, ids, rendezvous,
               dummy_float, wave_counts, graph_error,
               q.view(-1), k.view(-1), v.view(-1), valid.view(-1),
               q_pool.view(-1), k_pool.view(-1), v_pool.view(-1),
               scores.view(-1), mask.view(-1), output.view(-1),
               gate.view(-1), sm_count, stream=torch.cuda.current_stream())

    launch(probe)
    torch.cuda.synchronize()
    measured = ids.cpu().tolist()
    if len(set(measured)) != sm_count or min(measured) < 0 or max(measured) >= slots:
        raise RuntimeError("physical SM discovery failed")
    compile_stream = compile_vsa_full if full else compile_vsa_dynamic
    instructions = torch.tensor(compile_stream(
        policy["num_prefix_tiles"], start_tile, end_tile, keep, measured, slots
    ), device="cuda", dtype=torch.int32)
    mask.zero_()
    output.fill_(float("nan"))
    counts.zero_()
    ids.fill_(-1)
    wave_counts.zero_()
    graph_error.zero_()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    launch(instructions)
    end.record()
    torch.cuda.synchronize()
    if int(graph_error[0]) or not torch.equal(wave_counts, torch.full_like(wave_counts, sm_count)):
        raise RuntimeError("Hazy VSA wave barrier failed")
    if not torch.equal(counts, torch.full_like(counts, 2)):
        raise RuntimeError("Hazy VSA instruction stream did not complete")
    mask_mismatches = int((mask != expected_mask).count_nonzero())
    diff = (output.float() - expected.float()).abs()
    outside = int((~torch.isclose(output.float(), expected.float(),
                                  atol=0.03125, rtol=0.015625)).count_nonzero())
    print(json.dumps({"gpu_ms": start.elapsed_time(end),
                      "mask_mismatches": mask_mismatches,
                      "mask_elements": mask.numel(),
                      "attention_max_abs_error": float(diff.max()),
                      "attention_mean_abs_error": float(diff.mean()),
                      "attention_outside_tolerance": outside,
                      "attention_elements": output.numel()}), flush=True)
    if mask_mismatches or outside:
        raise AssertionError("one-launch Hazy dynamic VSA differs from FastVideo")
    print("PASS: Hazy one-launch dynamic BF16 VSA " +
          ("with compression" if full else "fine attention"), flush=True)


if __name__ == "__main__":
    main()
