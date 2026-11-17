# Hazy step-0 block-0 Q/K norm, RoPE, and VSA tiling gate

## Recipe and capture

- Checkpoint: `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer`, revision
  `9d0d0d3ae69365aded83fab5562f00f9f06c12aa`.
- FastVideo source: `ce970eb715215c79a169658d7aa6c8a96e64891e`.
- Transformer linears: NVFP4. Attention: BF16 sparse VSA,
  `FASTVIDEO_H3_VSA_FP4=0`, `FASTVIDEO_VSA_PTX_BLACKWELL=0`.
- Prompt 0, seed 1101, 1344 × 768, 124 requested frames, eight transformer
  forwards. The first VSA call is step 0, block 0.
- Exact boundary capture: Modal volume
  `h3-megakernel-cache/runs/attention-boundary-20261116-r2/`,
  [capture run](https://modal.com/apps/r19morris/main/ap-sJEBSllo0SN0eP5Z8lC6yH).
  `manifest.json` is verified and hashes every saved tensor.
- Frozen comparison: `h3-megakernel-cache/runs/reference-bf16-vsa-20261115/`.

Reproduce the capture and combined gate from the repository root with a fresh
capture run ID (Modal volume run directories are immutable):

```sh
H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo .venv/bin/modal run \
  baseline/fastvideo_modal.py --run-id <fresh-run-id> \
  --script projection_capture.py --args=--attention-inputs
H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo .venv/bin/modal run \
  megakernel/cutlass_probe_modal.py --mode attention-boundary \
  --capture-run-id <fresh-run-id>
H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo .venv/bin/modal run \
  megakernel/cutlass_probe_modal.py --mode projection-attention \
  --capture-run-id <fresh-run-id>
```

The captured pre-tile Q/K/V/gate tensors are `[1, 37771, 56, 128]` BF16.
The tile map places 37,771 real tokens into 668 tiles of 64 rows, for
42,752 padded rows. The captured Q/K norm weights have 128 channels and the
BF16 RoPE tables have 96 channels per token.

## Results

The capture-to-reference geometry check scattered each pre-tile tensor with
`untile_combined_index` and compared all elements, including zero padding.
Q, K, V, and gate each had **zero BF16 mismatches** with the frozen VSA input.
[Boundary check run](https://modal.com/apps/r19morris/main/ap-xb9hN5W06IlrmkdR4o8Tkx).

The new Hazy opcode 6 uses all twelve CTA warps. Each warp computes one
128-channel head's RMS norm, applies the 96-channel partial RoPE to Q and K,
and optionally writes into the padded tile layout while copying V and gate.
It ran on all 188 physical SMs in the Hazy persistent grid. Three repeats
were deterministic. Across 270,742,528 elements per tensor:

The isolated gate fed the operation saved projection tensors. A subsequent
eight-instruction Hazy stream chained smoke, four NVFP4 projections, a
cross-SM release/acquire barrier, tiled Q/K norm/RoPE, and smoke in **one
launch**. Its four raw projections were bit-exact to FastVideo, V and gate
were bit-exact after tiling, and Q/K had the same differences and tolerance
as the isolated gate on all three repeats. The barrier reached all 188 SMs
without a timeout or dependency error. [Chained GPU run](https://modal.com/apps/r19morris/main/ap-VVpFVuI6oHKryGlXj28Muy).

| Output | BF16 mismatches | Max absolute error | Outside tolerance |
| --- | ---: | ---: | ---: |
| Q, pre-tile and tiled | 3,887 | 0.03125 | 0 |
| K, pre-tile and tiled | 3,691 | 0.0625 | 0 |
| V, tiled | 0 | 0 | 0 |
| Gate, tiled | 0 | 0 | 0 |

Tolerance for Q/K was `atol=0.03125, rtol=0.015625` against FastVideo's
fused Triton norm/RoPE. The sparse Q/K differences are consistent with BF16
rounding differences;
this gate does **not** claim bit-exact Q/K or downstream attention parity.
The V and gate scatter is bit-exact. [Hazy GPU run](https://modal.com/apps/r19morris/main/ap-doGw1U1MCCwBdnjICtkZkr).
The earlier four-projection stream also passed its three-repeat bit-exact
regression after this operation was added:
[regression run](https://modal.com/apps/r19morris/main/ap-4WClCwxyxrv2jW1mvZndAk).

## Remaining milestone-3 work

The exact per-head VSA block mask and BF16 sparse attention/compression are
still missing from Hazy. Then output/MLP linears and residuals must form a
complete block, 50 blocks must match for one step, and all eight steps must
produce matching video and audio latents. No generation FPS is available from
Hazy until that complete path runs.
