"""Compare Hazy pooled Q/K scores against the exact FastVideo capture."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_boundary import read_tensor
from compiler import compile_vsa_scores

import torch
from mk_h3_projection import run_vsa_scores


def main() -> None:
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 required")
    root = Path(os.environ["H3MK_REFERENCE_CAPTURE"])
    manifest = json.loads((root / "reference_manifest.json").read_text())
    if not manifest.get("verified") or manifest["env"]["FASTVIDEO_H3_VSA_FP4"] != "0":
        raise RuntimeError("verified BF16 VSA capture required")
    q_pool = read_tensor(root, manifest, "step0_block0_q_pooled").cuda()
    k_pool = read_tensor(root, manifest, "step0_block0_k_pooled").cuda()
    expected = read_tensor(root, manifest, "step0_block0_vsa_scores").cuda()
    actual = torch.empty_like(expected)
    assert q_pool.shape == k_pool.shape == (1, 56, 668, 128)
    assert actual.shape == (1, 56, 668, 668)
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
        run_vsa_scores(instructions, timings, counts, ids, rendezvous,
                       dummy_float, dummy_int, dummy_int,
                       q_pool.view(-1), k_pool.view(-1), actual.view(-1),
                       sm_count, stream=torch.cuda.current_stream())

    launch(probe)
    torch.cuda.synchronize()
    measured = ids.cpu().tolist()
    if len(set(measured)) != sm_count or min(measured) < 0 or max(measured) >= slots:
        raise RuntimeError("physical SM discovery failed")
    instructions = torch.tensor(compile_vsa_scores(56, measured, slots),
                                device="cuda", dtype=torch.int32)
    for rep in range(2):
        actual.fill_(float("nan"))
        counts.zero_()
        ids.fill_(-1)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        launch(instructions)
        end.record()
        torch.cuda.synchronize()
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise RuntimeError("Hazy VSA scores instruction did not complete")
        diff = (actual - expected).abs()
        outside = int((~torch.isclose(actual, expected, atol=1e-4, rtol=1e-4)).count_nonzero())
        print(json.dumps({"rep": rep, "gpu_ms": start.elapsed_time(end),
                          "max_abs_error": float(diff.max()),
                          "mean_abs_error": float(diff.mean()),
                          "outside_tolerance": outside,
                          "elements": actual.numel()}), flush=True)
        if outside:
            raise AssertionError("Hazy VSA scores differ from FastVideo")
    print("PASS: Hazy tile-64 VSA pooled scores", flush=True)


if __name__ == "__main__":
    main()
