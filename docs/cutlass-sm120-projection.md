# CUTLASS NVFP4 projection probe on RTX PRO 6000

## Scope

This is a standalone feasibility probe for the first DiT projection in the
FastVideo no-FP4-attention recipe. The checkpoint remains
`FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer` at
`9d0d0d3ae69365aded83fab5562f00f9f06c12aa`, with FastVideo
`ce970eb715215c79a169658d7aa6c8a96e64891e`, SM120, and
`FASTVIDEO_H3_VSA_FP4=0`. This setting keeps the transformer NVFP4 linears;
it selects BF16 arithmetic for VSA attention.

## What was implemented

- Pinned CUTLASS v4.4.2 at `da5e086dab31d63815acafdac9a9c5893b1c69e2`
  through `megakernel/scripts/bootstrap_cutlass.py`. The checkout is ignored
  by Git. The probe adapts [NVIDIA's SM120 example 79a](https://github.com/NVIDIA/cutlass/blob/v4.4.2/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu).
- Added `megakernel/src/cutlass_projection.cu`: one standalone NVFP4 × NVFP4
  block-scaled GEMM with BF16 output, a 128×128×128 tile, 1-CTA cluster, and
  two mainloop pipeline stages. Its C++ binding accepts preallocated packed
  operands, scale tensors, output, and workspace; it exposes CUTLASS's CTA
  thread count, shared-storage size, workspace size, and grid shape.
- Added `megakernel/tests/projection_parity.py`: select the first packed
  `transformer_blocks.*.attn.to_q` weight in the pinned export, quantize 128
  rows and then all 37,771 rows from the exact captured projection input with FastVideo's
  128×4 scale layout, and compare CUTLASS's output to FastVideo's `mm_fp4`
  result. The test checks
  packed K, reports numeric error, and rejects a collective whose shared
  storage cannot coexist with Hazy's static control state.
- Added `megakernel/cutlass_probe_modal.py` to inspect the checkpoint export
  and run the build and parity test on one RTX PRO 6000 when cloud execution
  is authorized. The successful run is linked in
  `results/cutlass-projection-sm120-20261116.md`.

The original feasibility run used a block input before AdaLN and normalization.
The later exact-input run and Hazy integration are documented in
`results/hazy-projection-gate-20261116.md`.

## Runtime fit and integration

The later Hazy integration and exact-input parity passed; see
`results/hazy-projection-gate-20261116.md`. The discussion below records the
resource analysis made before that integration.

The selected CUTLASS SM120 kernel has 8 MMA warps plus one 4-warp loader group:
384 threads, matching the Hazy SM120 CTA. This follows from
`MaxThreadsPerBlock` in the [pinned CUTLASS SM120 kernel source](https://github.com/NVIDIA/cutlass/blob/v4.4.2/include/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp). Thread count
alone does not establish that the kernels compose. CUTLASS's body expects
control of all 12 warps and its own shared storage, while Hazy dispatches
those warps as controller, loader, launcher, storer, and eight consumers.

Hazy's current 100,000-byte CTA allocation includes 10,496 static bytes. Its
dynamic page base is aligned upward to 1,024 bytes, leaving at most 88,480
bytes for an aliased CUTLASS scratch region. The measured CUTLASS
`SharedStorage` is 51,200 bytes, leaving 37,280 bytes of margin. Its workspace
size is zero. This two-stage tile therefore needs no change to Hazy's current
shared-memory budget. The pinned CUTLASS kernel declares 40 registers for the
loader group and 232 for MMA groups; composed register usage was not measured.
The Hazy instruction now uses an all-warp task boundary, a static persistent
scheduler for 188 CTAs, and exclusive page ownership. The full projection has
16,576 tiles. `compiler.py` declares this instruction's resources separately
from the synthetic graph opcodes.

## Validation

After the user authorized uploading the local source to Modal, the extension
compiled and ran on one RTX PRO 6000 Blackwell (SM120). It selected
`transformer_blocks.0.attn.to_q` with packed weight shape `[7168, 2688]`,
corresponding to K=5376. Both the 128-row tile and the real 37,771-row
sequence matched FastVideo's BF16 output bit for bit: maximum and mean absolute
error were zero. The 37,771-row case used A scales padded to 37,888 rows and
CUTLASS grid `(56, 296)`. The check passed the non-multiple-of-128 tail.
See the [Modal run](https://modal.com/apps/r19morris/main/ap-66JE7bAGA5gTk0RX7VM7FS)
and `results/cutlass-projection-sm120-20261116.md`.

To repeat the standalone probe:

```sh
H3MK_FV_SRC=/absolute/path/to/FastVideo \
  .venv/bin/modal run megakernel/cutlass_probe_modal.py \
  --mode verify --capture-run-id projection-exact-20261116-r3
```

The standalone pass gate is met: `can_implement=true`, 384 threads, 51,200
shared bytes, zero workspace, and exact checkpoint-weight output parity. The
subsequent exact-input and Hazy runs passed as recorded in
`results/hazy-projection-gate-20261116.md`.
