# Milestone 3: what works and what remains

Milestone 3 means the **entire eight-step DiT** produces video and audio
latents that match the pinned FastVideo reference. It does not include the
text encoder or decoders; those are milestone 4. The current Hazy code has
validated the first four matrix multiplications, Q/K normalization, partial
RoPE, the VSA tile layout, and BF16 sparse fine attention with a frozen mask
in block 0, step 0. It cannot
yet generate a video or report generation throughput in frames per second.
The reference request sets the output video's playback rate to 24 fps; that
is separate from how quickly a GPU generates its frames.
The full target runs 50 blocks per step, or 400 block invocations.

## The path through one DiT block

```text
packed text + video + audio tokens
        │
        ▼
RMSNorm + AdaLN modulation
        │
        ├── Q projection ─┐
        ├── K projection ─┼── Q/K norm + RoPE ── VSA tile layout + mask
        ├── V projection ─┤                         │
        └── gate projection ┘                  sparse BF16 attention
                                                   │
                                         output projection + residual
                                                   │
                                         norm + MLP + residual
                                                   │
                                           next block (50 total)
                                                   │
                                       latent update (8 steps total)
```

The four projections have passed together in one Hazy launch at the full
37,771-row shape, with bit-exact BF16 parity on three repeated launches. The
compression gate is active in this checkpoint. See
`results/hazy-qkv-gate-20261116.md` for the capture, hashes, numerical results,
and GPU run links. The earlier single-Q result is in
`results/hazy-projection-gate-20261116.md`.

The next Hazy operation now normalizes and rotates Q/K and places Q/K/V/gate
into the 668 padded VSA tiles. It has also passed chained to all four
projections in one Hazy launch, with a cross-SM barrier between stages.
V and gate match the frozen BF16 reference
bit for bit. Q and K have 3,887 and 3,691 BF16 differences respectively out
of 270,742,528 elements each; all are within `atol=0.03125,
rtol=0.015625` across three repeated GPU runs. The tile mapping itself was
verified bit for bit using FastVideo's pre-tile tensors. See
`results/hazy-qknorm-tile-gate-20261116.md` for exact results and run links.

The BF16 sparse fine-attention calculation now also runs as a Hazy CUDA
instruction. Across the entire first attention call, its 306,446,336 output
elements were within the stated BF16 tolerance of FastVideo's Triton result
on two repeats, both alone and chained after the projections and tile stage
in one Hazy launch. That earlier check supplied the frozen exact block mask.
Dynamic mask construction and the trained coarse compression gate now also
pass for step 0 block 0 in one Hazy launch using captured Q/K/V/gate inputs.
The mask matches all 24,988,544 entries exactly, and the post-compression
output stays within BF16 tolerance on all 306,446,336 elements. The launch
takes about 4.45 seconds, so it needs a faster fine-attention body before the
full 400-block run. See `results/hazy-dynamic-vsa-gate-20261116.md`.

## Why the rest is substantial

The reference's attention call rearranges 37,771 real rows into 42,752
tile-ordered rows, including padding. At step 0, block 0, it passes a
`[1, 56, 668, 668]` per-head VSA mask and 668 variable-length entries to
the sparse attention kernel. Q/K normalization, partial RoPE, tile order,
top-k mask selection, prefix exemptions, dense step/layer policy, and the
trained gate all affect the answer. The exact BF16 attention implementation
then feeds output projection, residuals, normalization, and MLP linears.

FastVideo's tile-64 reference uses a Triton sparse-attention kernel launched
from the host. The Hazy fine-attention operation is a CUDA rewrite inside the
persistent launch. Hazy now also generates the mask and runs the compression
branch for the first block, using captured Q/K/V/gate inputs. The output
projection, residual, and MLP still need to join the same launch before a
complete block can be claimed.

## Pass gates

| Gate | Current status |
| --- | --- |
| Exact reference and first Q projection | Passed |
| Q/K/V and active gate in one Hazy launch | Passed |
| Q/K norm and partial RoPE | Passed within stated BF16 tolerance |
| VSA tile layout and V/gate scatter | Passed; V/gate bit-exact |
| BF16 sparse fine attention with frozen mask | Passed within stated BF16 tolerance |
| Dynamic VSA mask and trained compression branch, block 0 step 0 | Passed with captured Q/K/V/gate inputs in one launch |
| Output/MLP linears and complete block | Open |
| All 50 blocks for one step | Open |
| Eight steps and matching final video/audio latents | Open — milestone 3 pass condition |

The next validation target is to chain the projections to dynamic VSA, then
untile and quantize its output for `to_out` and complete the block. The saved
tensors in `h3-megakernel-cache/runs/reference-vsa-scores-20261116-a/` are the comparison
contract. A block-level mismatch must be resolved before running 50 blocks
or eight steps.
