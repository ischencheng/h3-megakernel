"""Time the tensor-core fine attention at the full first-recipe shape.

Synthetic Q/K/V with an 80%-sparse exempt-prefix mask (8 prefix tiles, 132 of
660 video tiles per video query). The diagnostic stage mask in instruction
word 9 removes one stage at a time; it never appears in a compiled recipe.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

import h3_dit as hd
import mk_h3_dit


def main():
    import glob
    import subprocess
    path = glob.glob("/root/megakernel/build/RTXPRO6000/mk_h3_dit*.so")[0]
    print(subprocess.run(["cuobjdump", "-res-usage", path], text=True,
                         capture_output=True).stdout[-400:], flush=True)
    torch.manual_seed(0)
    heads, tiles, prefix, keep = 56, 668, 8, 132
    padded = tiles * 64
    q, k, v = ((torch.randn(padded, heads, 128, device="cuda") * 0.5).to(torch.bfloat16)
               for _ in range(3))
    valid = torch.full((tiles,), 64, device="cuda", dtype=torch.int64)
    mask = torch.zeros(heads, tiles, tiles, device="cuda", dtype=torch.uint8)
    mask[:, :prefix] = 1
    mask[:, :, :prefix] = 1
    choice = torch.rand(heads, tiles, tiles - prefix, device="cuda").topk(keep, dim=-1).indices + prefix
    mask.scatter_(-1, choice, 1)
    mask[:, :prefix] = 1
    pairs = int(mask.sum())
    out = torch.zeros_like(q)
    rows = {}
    cases = (("full", 0), ("no K stage", 1), ("no QK mma", 2), ("no V stage", 4), ("no exp", 8),
             ("no PV mma", 16), ("no stage, no mma", 1 | 2 | 4 | 16), ("mask scan only", 32),
             ("only K stage", 2 | 4 | 8 | 16), ("only V stage", 1 | 2 | 8 | 16),
             ("only QK mma", 1 | 4 | 8 | 16), ("only PV mma", 1 | 2 | 4 | 8),
             ("barriers only", 1 | 2 | 4 | 8 | 16), ("nothing", 1 | 2 | 4 | 8 | 16 | 64),
             ("no barriers: full", 64), ("no barriers: only QK mma", 64 | 1 | 4 | 8 | 16),
             ("no barriers: only PV mma", 64 | 1 | 2 | 4 | 8), ("no barriers: only exp", 64 | 1 | 2 | 4 | 16),
             ("no barriers: only stages", 64 | 2 | 8 | 16))
    for name, skip in cases:
        best = None
        for _ in range(2):
            p = hd.Program(mk_h3_dit)
            p.emit("attention", "SPARSE_ATTENTION_MMA", tiles, heads, p.buf(q), p.buf(k), p.buf(v),
                   p.buf(mask), p.buf(valid), p.buf(out), skip)
            ms = p.run()
            best = ms if best is None else min(best, ms)
        rows[name] = best
        print(f"BENCH {name}: {best:.1f} ms ({best * 1e3 / pairs * 564:.1f} us per tile pair per group)",
              flush=True)
    print(json.dumps({"tile_pairs": pairs, "ms": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
