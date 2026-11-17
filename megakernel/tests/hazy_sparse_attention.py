"""Compare one Hazy BF16 sparse-attention instruction with FastVideo Triton."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_boundary import read_tensor
from compiler import compile_sparse_attention

import torch
from fastvideo_kernel.block_sparse_attn import block_sparse_attn
from mk_h3_projection import run_sparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-row", type=int, default=0)
    parser.add_argument("--row-count", type=int, default=64)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("RTX PRO 6000 SM120 required")
    root = Path(os.environ.get("H3MK_REFERENCE_CAPTURE",
                               "/cache/runs/reference-bf16-vsa-20261115"))
    manifest = json.loads((root / "reference_manifest.json").read_text())
    if not manifest.get("verified") or manifest["env"]["FASTVIDEO_H3_VSA_FP4"] != "0":
        raise RuntimeError("missing verified BF16 VSA reference")
    q, k, v = (read_tensor(root, manifest, f"step0_block0_{name}").cuda()
               for name in ("q", "k", "v"))
    mask = read_tensor(root, manifest, "step0_block0_vsa_mask").cuda()
    valid = read_tensor(root, manifest, "step0_block0_valid_lengths").cuda()
    if q.shape != (1, 42752, 56, 128) or mask.shape != (1, 56, 668, 668):
        raise RuntimeError("unexpected first-block sparse-attention geometry")
    expected_bhsd, _ = block_sparse_attn(
        q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(), mask, valid)
    expected = expected_bhsd.transpose(1, 2).contiguous()
    output = torch.empty_like(q)
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
        run_sparse(instructions, timings, counts, ids, rendezvous,
                   dummy_float, dummy_int, dummy_int,
                   q.view(-1), k.view(-1), v.view(-1), mask.view(-1),
                   valid.view(-1), output.view(-1), sm_count,
                   stream=torch.cuda.current_stream())

    launch(probe)
    torch.cuda.synchronize()
    measured = ids.cpu().tolist()
    if len(set(measured)) != sm_count or min(measured) < 0 or max(measured) >= slots:
        raise RuntimeError("physical SM discovery failed")
    instructions = torch.tensor(compile_sparse_attention(
        args.first_row, args.row_count, measured, slots), device="cuda", dtype=torch.int32)
    rows = torch.arange(args.first_row, args.first_row + args.row_count, device="cuda")
    heads, seq = rows // 42752, rows % 42752
    for rep in range(2):
        output.fill_(float("nan"))
        counts.zero_()
        ids.fill_(-1)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        launch(instructions)
        end.record()
        torch.cuda.synchronize()
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise RuntimeError("Hazy sparse-attention instruction did not complete")
        actual_rows = output[0, seq, heads]
        expected_rows = expected[0, seq, heads]
        diff = (actual_rows.float() - expected_rows.float()).abs()
        outside = int((~torch.isclose(actual_rows.float(), expected_rows.float(),
                                      rtol=0.015625, atol=0.03125)).count_nonzero())
        mismatch = int((actual_rows != expected_rows).count_nonzero())
        print(json.dumps({"rep": rep, "first_row": args.first_row,
                          "row_count": args.row_count, "gpu_ms": start.elapsed_time(end),
                          "max_abs_error": float(diff.max()),
                          "mean_abs_error": float(diff.mean()),
                          "mismatched_elements": mismatch,
                          "outside_tolerance": outside,
                          "elements": actual_rows.numel()}), flush=True)
        if outside:
            worst = int(diff.reshape(-1).argmax())
            local_row, channel = divmod(worst, 128)
            global_row = args.first_row + local_row
            head, query = divmod(global_row, 42752)
            selected = mask[0, head, query // 64].nonzero().flatten().tolist()
            key_rows = [tile * 64 + offset for tile in selected
                        for offset in range(int(valid[tile]))]
            key_matrix = k[0, key_rows, head].float()
            value_matrix = v[0, key_rows, head].float()
            scores = key_matrix @ q[0, query, head].float() * (128 ** -0.5)
            manual = torch.softmax(scores, dim=0) @ value_matrix
            print(json.dumps({"worst_row": global_row, "head": head, "query": query,
                              "channel": channel, "selected_keys": len(key_rows),
                              "hazy_value": float(actual_rows[local_row, channel]),
                              "triton_value": float(expected_rows[local_row, channel]),
                              "fp32_manual_value": float(manual[channel]),
                              "q_abs_max": float(q[0, query, head].float().abs().max()),
                              "key_abs_max": float(key_matrix.abs().max())}), flush=True)
            raise AssertionError("Hazy BF16 sparse attention exceeds tolerance")
    print("PASS: Hazy single-kernel BF16 sparse-attention gate", flush=True)


if __name__ == "__main__":
    main()
