# First FastH3 projection inside Hazy on SM120

Status: **one-projection gate passed** on 2026-11-16. This is the first
`transformer_blocks.0.attn.to_q` linear, not a complete DiT block.

## Pinned reference and exact capture

- FastVideo checkout: `ce970eb715215c79a169658d7aa6c8a96e64891e`.
- Checkpoint: `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer`, revision
  `9d0d0d3ae69365aded83fab5562f00f9f06c12aa`.
- RTX PRO 6000 Blackwell (SM120); 1344×768, 124 frames, prompt index 0 from
  `baseline/prompts.json` (SHA256
  `8a0d0914189356d56447b166cd102d0fe72b131d5ee1bea04baa58e9df218f70`),
  seed 1101, eight denoising forwards (`num_inference_steps=9`).
- NVFP4 transformer linears, `FASTVIDEO_H3_VSA_FP4=0`, BF16 VSA,
  `VIDEO_SPARSE_ATTN_H3` at 80% sparsity and tile 64. ConvRot is not involved
  in this projection.
- [Successful capture run](https://modal.com/apps/r19morris/main/ap-yscStTvcbV4jvX9lN1muuE),
  volume path `/cache/runs/projection-exact-20261116-r3/`, `returncode=0`.
  The verified manifest is at that path's `manifest.json`.

The capture intercepted `_shared_input_projections` at step 0 for
`minimax_h3.transformer_blocks.0.attn.to_q`, after block RMSNorm and AdaLN.
It saved the BF16 input `[1, 37771, 5376]` with stride
`[203056896, 5376, 1]` and SHA256
`2d3cfe899edae74256ef39646e4f5a1590c28a0eb2b6d55f6ca7765ee55dec1a`,
and BF16 output `[1, 37771, 7168]` with stride
`[270742528, 7168, 1]` and SHA256
`efeedbd9094ea6112f7e6d08344c5b91014564055ec7e6f419e99043c6d52e72`.
The manifest records checkpoint IDs, step, prompt, seed, schedule, environment,
shapes, dtypes, source/stored strides, and file hashes. The parity probes
verify the manifest, `returncode`, and both file hashes before using it.

Two earlier capture attempts were rejected as references: `r1` missed the
qualified layer prefix; `r2` saved tensors but FastVideo wrapped the intentional
stop signal and returned 1. Neither has a verified manifest.

## Arithmetic parity

The [standalone parity run](https://modal.com/apps/r19morris/main/ap-m3J3DquzNY4GnPsxtY5OlN)
quantized the exact captured input with FastVideo's NVFP4 128×4 scale layout
and used the pinned packed weight, scale, and alpha. FastVideo `mm_fp4` matched
the captured output bit for bit. The standalone CUTLASS output matched both
at 128 rows and at all 37,771 rows, including the partial final tile.
Maximum error, mean error, and mismatch count were all zero with
`atol=rtol=0` over 270,742,528 full-shape BF16 elements.

The [Hazy instruction run](https://modal.com/apps/r19morris/main/ap-876IU3SF6vJTjSpmPCbgxW)
used 188 resident CTAs, 384 threads per CTA, eight consumer warps plus four
control/worker warps, and CUTLASS static persistent scheduling. The first
attempt with CUTLASS's default SM120 scheduler wrote only one 128×128 tile;
it was rejected. The static scheduler advances each CTA by the 188-CTA grid
size and covers 16,576 tiles exactly once, including the 11-row final tile.
The test prefills output with NaNs to catch missed writes and runs a Hazy smoke
instruction both before and after the GEMM. The final run matched the captured
output bit for bit at 128 and 37,771 rows on **three launches each**:
max error 0, mean error 0, mismatch count 0. All 188 physical SM IDs were
unique and the surrounding smoke instructions completed twice per CTA.

The CUTLASS collective uses 51,200 shared bytes and no global workspace.
Hazy allocates 100,000 total shared bytes per CTA, including 10,496 static
bytes, leaving at least 88,480 aligned dynamic bytes. The projection owns the
five aliased Hazy pages between all-warp entry and exit barriers. The host
prepares a 2,304-byte CUTLASS parameter record before launch; there are no
device-side launches or allocations. CUTLASS declares 232 registers for its
MMA groups and 40 for its loader group; composed register usage was not
separately measured.

The [post-change SM120 regression](https://modal.com/apps/r19morris/main/ap-eTUPb1B3RXzH6RabgP1izs)
also passed the existing mixed-opcode/ring-reuse smoke and five-wave graph
tests, including cross-SM dependencies, arena reuse, and repeated launches.

## Commands

```sh
H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo \
  .venv/bin/modal run baseline/fastvideo_modal.py \
  --run-id projection-exact-20261116-r3 --script projection_capture.py

H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo \
  .venv/bin/modal run megakernel/cutlass_probe_modal.py \
  --mode verify --capture-run-id projection-exact-20261116-r3

H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo \
  .venv/bin/modal run megakernel/cutlass_probe_modal.py \
  --mode hazy --capture-run-id projection-exact-20261116-r3
```

The remaining milestone-3 gates are K/V and other linears, exact-mask BF16
VSA, a complete block, 50 blocks for one step, and all eight steps. This
projection result does not establish those gates.

