"""Probe FlashInfer's SM120 BF16 tile-64 VSA against pinned FastH3 tensors.

This is a host-launched comparison target. It is not a Hazy device operation.
"""

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_boundary import read_tensor

import torch
from flashinfer import BlockSparseAttentionWrapper


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "outside_tolerance": int((~torch.isclose(
            actual.float(), expected.float(), atol=0.03125, rtol=0.015625
        )).count_nonzero()),
        "elements": actual.numel(),
    }


def main() -> None:
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 required")
    root = Path(os.environ.get(
        "H3MK_REFERENCE_CAPTURE", "/cache/runs/reference-bf16-vsa-20261115"
    ))
    manifest = json.loads((root / "reference_manifest.json").read_text())
    if not manifest.get("verified") or manifest["env"]["FASTVIDEO_H3_VSA_FP4"] != "0":
        raise RuntimeError("verified BF16 VSA capture required")

    q, k, v = (read_tensor(root, manifest, f"step0_block0_{name}").cuda()
               for name in ("q", "k", "v"))
    mask = read_tensor(root, manifest, "step0_block0_vsa_mask").cuda().bool()
    valid = read_tensor(root, manifest, "step0_block0_valid_lengths").cuda().int()
    assert q.shape == k.shape == v.shape == (1, 42752, 56, 128)
    assert mask.shape == (1, 56, 668, 668)
    assert valid.shape == (668,)

    # vLLM-Omni's H3 provider preserves the logical tile-64 mask and excludes
    # zero-length KV blocks. The probe uses exactly one head and one query tile
    # to keep the FP32 oracle small while retaining the full KV span.
    head = int(os.environ.get("H3MK_PROBE_HEAD", "0"))
    qtile = int(os.environ.get("H3MK_PROBE_QTILE", "0"))
    selected = mask[0, head, qtile] & (valid > 0)
    local_mask = selected[None, None, :].contiguous()
    block_map = selected[None, None, None, :].contiguous()
    qh = q[0, qtile * 64:(qtile + 1) * 64, head:head + 1].contiguous()
    kh = k[0, :, head:head + 1].contiguous()
    vh = v[0, :, head:head + 1].contiguous()
    block_ids = selected.nonzero().flatten()
    key_ids = torch.cat([
        tile * 64 + torch.arange(int(valid[tile]), device="cuda")
        for tile in block_ids.tolist()
    ])
    if not key_ids.numel():
        raise RuntimeError("sample selected no valid keys")
    scores = (qh[:, 0].float() @ kh[key_ids, 0].float().T) * (128 ** -0.5)
    oracle = (scores.softmax(-1) @ vh[key_ids, 0].float()).bfloat16()
    print(json.dumps({"flashinfer_version": getattr(sys.modules["flashinfer"], "__version__", "unknown"),
                      "head": head, "query_tile": qtile,
                      "selected_blocks": int(block_ids.numel()),
                      "partial_blocks": int(((valid < 64) & selected).sum()),
                      "zero_length_blocks": int((valid == 0).sum()),
                      "selected_keys": int(key_ids.numel())}), flush=True)

    # vLLM-Omni calls this lower-level FlashInfer entry point. Unlike the
    # BlockSparseAttentionWrapper, it passes the exact valid lengths to the
    # SM120 kernel. Keep the metadata conversion identical to its adapter.
    try:
        from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_fwd
        key_blocks = block_map.shape[-1]
        ids = torch.arange(key_blocks, device="cuda", dtype=torch.int32)
        indices = torch.where(block_map, ids, key_blocks).sort(dim=-1).values
        indices = torch.where(indices == key_blocks, -1, indices).contiguous()
        counts = block_map.sum(dim=-1, dtype=torch.int32).contiguous()
        def call():
            return bsa_attn_sm120_blk64_fwd(
                qh.unsqueeze(0), kh.unsqueeze(0), vh.unsqueeze(0),
                indices, indices.shape[-1], block_sizes=valid,
                q2k_block_nums=counts, softmax_scale=128 ** -0.5,
            )

        call()  # Compile and warm the kernel before the measured launch.
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result, _ = call()
        end.record()
        torch.cuda.synchronize()
        metrics = compare(result[0, :, 0], oracle)
        print(json.dumps({"backend": "vllm_omni_direct_sm120", "status": "ok",
                          "gpu_ms": start.elapsed_time(end), **metrics}), flush=True)
        if metrics["outside_tolerance"]:
            raise AssertionError("direct SM120 result exceeds BF16 tolerance")
        if os.environ.get("H3MK_COMPARE_FASTVIDEO") == "1":
            from fastvideo_kernel.block_sparse_attn import block_sparse_attn

            fastvideo, _ = block_sparse_attn(
                q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(), mask, valid.to(torch.int64)
            )
            reference = fastvideo[0, head, qtile * 64:(qtile + 1) * 64]
            parity = compare(result[0, :, 0], reference)
            print(json.dumps({"backend": "fastvideo_triton_comparison", **parity}), flush=True)
            if parity["outside_tolerance"]:
                raise AssertionError("direct SM120 result differs from FastVideo Triton")
    except Exception as exc:
        print(json.dumps({"backend": "vllm_omni_direct_sm120", "status": "error",
                          "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        raise

    # The explicit SM120 backend does not document kv_block_lens. Its result
    # is a valid comparison only if all selected KV blocks are full length.
    for backend in ("vsa_sm120_blk64", "cake"):
        if backend == "vsa_sm120_blk64" and bool(((valid < 64) & selected).any()):
            print(json.dumps({"backend": backend, "status": "skipped_partial_kv"}), flush=True)
            continue
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        try:
            wrapper = BlockSparseAttentionWrapper(workspace, backend=backend)
            kwargs = dict(indptr=None, indices=None, M=64, N=42752, R=64, C=64,
                          num_qo_heads=1, num_kv_heads=1, head_dim=128,
                          q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                          o_data_type=torch.bfloat16, block_mask=local_mask)
            if backend == "cake":
                kwargs["kv_block_lens"] = valid.clamp(min=1)
            wrapper.plan(**kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = wrapper.run(qh, kh, vh)
            torch.cuda.synchronize()
            print(json.dumps({"backend": backend, "status": "ok",
                              "wall_s": time.perf_counter() - start,
                              **compare(result[:, 0], oracle)}), flush=True)
        except Exception as exc:
            print(json.dumps({"backend": backend, "status": "error",
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
