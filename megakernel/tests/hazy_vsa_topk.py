"""Compare Hazy dynamic VSA selection with the pinned FastVideo mask."""

import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_boundary import read_tensor
from compiler import compile_vsa_topk

import torch
from mk_h3_projection import run_vsa_topk


def main() -> None:
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 required")
    root = Path(os.environ["H3MK_REFERENCE_CAPTURE"])
    manifest = json.loads((root / "reference_manifest.json").read_text())
    if not manifest.get("verified") or manifest["env"]["FASTVIDEO_H3_VSA_FP4"] != "0":
        raise RuntimeError("verified BF16 VSA capture required")
    policy = manifest["step0_block0_vsa_policy"]
    if not policy["exempt"] or len(policy["video_tile_spans"]) != 1:
        raise RuntimeError("this Hazy VSA selector supports one exempt video region")
    start_tile, end_tile = policy["video_tile_spans"][0]
    keep = math.ceil((1 - policy["span_sparsities"][0]) * (end_tile - start_tile))
    scores = read_tensor(root, manifest, "step0_block0_vsa_scores").cuda()
    expected = read_tensor(root, manifest, "step0_block0_vsa_mask").cuda()
    actual = torch.empty_like(expected)
    assert scores.shape == expected.shape == (1, 56, 668, 668)
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

    def launch(instructions):
        run_vsa_topk(instructions, timings, counts, ids, rendezvous,
                     dummy_float, dummy_int, dummy_int,
                     scores.view(-1), actual.view(-1), sm_count,
                     stream=torch.cuda.current_stream())

    launch(probe)
    torch.cuda.synchronize()
    measured = ids.cpu().tolist()
    if len(set(measured)) != sm_count or min(measured) < 0 or max(measured) >= slots:
        raise RuntimeError("physical SM discovery failed")
    instructions = torch.tensor(compile_vsa_topk(
        56, policy["num_prefix_tiles"], start_tile, end_tile, keep, measured, slots
    ), device="cuda", dtype=torch.int32)
    for rep in range(2):
        actual.zero_()
        counts.zero_()
        ids.fill_(-1)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        launch(instructions)
        end.record()
        torch.cuda.synchronize()
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise RuntimeError("Hazy VSA top-k instruction did not complete")
        mismatch = actual != expected
        n = int(mismatch.count_nonzero())
        print(json.dumps({"rep": rep, "gpu_ms": start.elapsed_time(end),
                          "mask_mismatches": n, "selected_blocks": int(actual.sum()),
                          "elements": actual.numel(), "keep_per_video_row": keep}), flush=True)
        if n:
            position = mismatch.nonzero()[0].tolist()
            print(json.dumps({"first_mismatch": position,
                              "score": float(scores[tuple(position)]),
                              "actual": bool(actual[tuple(position)]),
                              "expected": bool(expected[tuple(position)])}), flush=True)
            raise AssertionError("Hazy VSA selection differs from FastVideo")
    print("PASS: Hazy dynamic tile-64 VSA top-k mask", flush=True)


if __name__ == "__main__":
    main()
