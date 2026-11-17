"""Verify captured FastH3 pooled scores and dynamic tile-64 selection."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_boundary import read_tensor

import torch
from fastvideo.attention.backends.video_sparse_attn_h3 import _build_block_mask, _pool_tiles


def main() -> None:
    root = Path(os.environ["H3MK_REFERENCE_CAPTURE"])
    manifest = json.loads((root / "reference_manifest.json").read_text())
    if not manifest.get("verified") or manifest["env"]["FASTVIDEO_H3_VSA_FP4"] != "0":
        raise RuntimeError("verified BF16 VSA capture required")
    policy = manifest["step0_block0_vsa_policy"]
    q = read_tensor(root, manifest, "step0_block0_q").cuda()
    k = read_tensor(root, manifest, "step0_block0_k").cuda()
    valid = read_tensor(root, manifest, "step0_block0_valid_lengths").cuda()
    expected_q_pool = read_tensor(root, manifest, "step0_block0_q_pooled").cuda()
    expected_k_pool = read_tensor(root, manifest, "step0_block0_k_pooled").cuda()
    expected_scores = read_tensor(root, manifest, "step0_block0_vsa_scores").cuda()
    expected_mask = read_tensor(root, manifest, "step0_block0_vsa_mask").cuda()
    tile = policy["tile_elems"]
    q_pool = _pool_tiles(q, valid, tile)
    k_pool = _pool_tiles(k, valid, tile)
    scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    sparsity = (0.0 if 0 in policy["dense_layers"] else policy["VSA_sparsity"])
    mask = _build_block_mask(
        scores, policy["num_prefix_tiles"], sparsity, policy["exempt"],
        tuple(tuple(span) for span in policy["video_tile_spans"]),
        tuple(policy["span_sparsities"]),
    )
    checks = {
        "q_pool_mismatches": int((q_pool != expected_q_pool).count_nonzero()),
        "k_pool_mismatches": int((k_pool != expected_k_pool).count_nonzero()),
        "scores_mismatches": int((scores != expected_scores).count_nonzero()),
        "mask_mismatches": int((mask != expected_mask).count_nonzero()),
        "selected_blocks": int(mask.sum()),
        "score_shape": list(scores.shape),
        "policy": policy,
    }
    row_scores = scores[0, 12, 12, 8:668]
    threshold = row_scores.topk(132).values[-1]
    tied = (row_scores == threshold).nonzero().flatten() + 8
    checks["boundary_tie_probe"] = {
        "threshold": float(threshold),
        "tied_columns": tied.tolist(),
        "selected_tied_columns": tied[expected_mask[0, 12, 12, tied]].tolist(),
    }
    print(json.dumps(checks), flush=True)
    if any(checks[name] for name in (
        "q_pool_mismatches", "k_pool_mismatches", "scores_mismatches", "mask_mismatches"
    )):
        raise AssertionError("captured VSA score or mask reconstruction differs")


if __name__ == "__main__":
    main()
