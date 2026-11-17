"""Compare Hazy tile-64 Q/K pooling against the captured FastVideo tensors."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_boundary import read_tensor
from compiler import compile_vsa_pool

import torch
from mk_h3_projection import run_vsa_pool


def main() -> None:
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 required")
    root = Path(os.environ["H3MK_REFERENCE_CAPTURE"])
    manifest = json.loads((root / "reference_manifest.json").read_text())
    if not manifest.get("verified") or manifest["env"]["FASTVIDEO_H3_VSA_FP4"] != "0":
        raise RuntimeError("verified BF16 VSA capture required")
    q = read_tensor(root, manifest, "step0_block0_q").cuda()
    k = read_tensor(root, manifest, "step0_block0_k").cuda()
    valid = read_tensor(root, manifest, "step0_block0_valid_lengths").cuda()
    expected_q = read_tensor(root, manifest, "step0_block0_q_pooled").cuda()
    expected_k = read_tensor(root, manifest, "step0_block0_k_pooled").cuda()
    q_pool, k_pool = torch.empty_like(expected_q), torch.empty_like(expected_k)
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
        run_vsa_pool(instructions, timings, counts, ids, rendezvous,
                     dummy_float, dummy_int, dummy_int,
                     q.view(-1), k.view(-1), valid.view(-1),
                     q_pool.view(-1), k_pool.view(-1), sm_count,
                     stream=torch.cuda.current_stream())

    launch(probe)
    torch.cuda.synchronize()
    measured = ids.cpu().tolist()
    if len(set(measured)) != sm_count or min(measured) < 0 or max(measured) >= slots:
        raise RuntimeError("physical SM discovery failed")
    instructions = torch.tensor(compile_vsa_pool(668, measured, slots),
                                device="cuda", dtype=torch.int32)
    for rep in range(2):
        q_pool.fill_(float("nan"))
        k_pool.fill_(float("nan"))
        counts.zero_()
        ids.fill_(-1)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        launch(instructions)
        end.record()
        torch.cuda.synchronize()
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise RuntimeError("Hazy VSA pooling instruction did not complete")
        metrics = {}
        for name, actual, expected in (("q", q_pool, expected_q), ("k", k_pool, expected_k)):
            diff = (actual - expected).abs()
            outside = int((~torch.isclose(actual, expected, atol=1e-5, rtol=1e-5)).count_nonzero())
            metrics[name] = {"max_abs_error": float(diff.max()),
                             "mean_abs_error": float(diff.mean()),
                             "outside_tolerance": outside,
                             "elements": actual.numel()}
            if outside:
                raise AssertionError(f"Hazy {name} pool differs from FastVideo")
        print(json.dumps({"rep": rep, "gpu_ms": start.elapsed_time(end),
                          "metrics": metrics}), flush=True)
    print("PASS: Hazy tile-64 VSA Q/K pooling", flush=True)


if __name__ == "__main__":
    main()
