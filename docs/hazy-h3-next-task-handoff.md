# Handoff: first FastH3 projection inside Hazy on RTX PRO 6000

**Completed 2026-11-16.** The exact capture, standalone parity, and full-shape
Hazy instruction passed. See `results/hazy-projection-gate-20261116.md` for
run links, hashes, resources, and numerical results. The sequence below is
preserved as the original task specification.

## Objective and fixed scope

Complete the **one-projection gate** in milestone 3 of
`docs/hazy-h3-plan.md`: execute the first DiT `attn.to_q` NVFP4 linear as a
Hazy instruction on one RTX PRO 6000 (SM120), then compare its BF16 output
with the exact FastVideo inference tensor. The first recipe is the FastVideo
FastH3 8-Step V2 NVFP4 Consumer checkpoint, revision
`9d0d0d3ae69365aded83fab5562f00f9f06c12aa`, with FastVideo revision
`ce970eb715215c79a169658d7aa6c8a96e64891e`. Use 1344×768, 124 frames,
prompt index 0 from `baseline/prompts.json`, seed 1101, and the eight-step
schedule (`num_inference_steps=9` in FastVideo). Set
`FASTVIDEO_H3_VSA_FP4=0`: transformer linears remain NVFP4, while VSA
attention uses regular BF16 arithmetic. Keep the checkpoint's sparse VSA
policy. ConvRot support is a later extension, outside this gate.

## State at handoff

- Milestones 1 and 2 are complete: SM120 Hazy smoke and a synthetic
  checkpoint-independent graph compiler passed. See
  `results/hazy-reference-sm120-20261115.md` and
  `results/hazy-compiler-sm120-20261116.md`.
- A **standalone** CUTLASS v4.4.2 SM120 NVFP4→BF16 `to_q` GEMM passed against
  FastVideo's `mm_fp4` bit for bit for 128 rows and the full 37,771 rows.
  The full GEMM has M=37,771, N=7,168, K=5,376, and 16,576 CUTLASS tiles.
  It uses 384 threads, 51,200 bytes of shared storage, and no global
  workspace. See `docs/cutlass-sm120-projection.md` and
  `results/cutlass-projection-sm120-20261116.md`.
- That standalone test used a captured **block input**, before AdaLN and
  normalization. It is not parity with the actual `attn.to_q` input. The
  CUTLASS body has not run inside Hazy.
- `baseline/projection_capture.py` was added to capture the exact step-0,
  block-0 `to_q` input/output by wrapping FastVideo's shared Q/K/V projection
  helper. **It has not been executed or validated.** Review its interception
  condition and generated manifest before relying on its tensors.
- The working tree contains uncommitted changes from prior milestones. Preserve
  them; inspect `git status --short` before editing.

The prior user explicitly authorized uploading the local FastVideo checkout
and this project to Modal for RTX PRO 6000 validation. The existing Modal
scripts build an image from those sources.

## Work sequence

### 1. Capture the exact reference projection

Inspect `baseline/projection_capture.py` against FastVideo's current
`MiniMaxH3Attention.forward` and `_shared_input_projections`. The helper can
prequantize a shared Q/K/V input and call the quantization method directly, so
a hook on `to_q.forward` alone may miss the projection. Confirm that the
wrapper identifies step 0 and `transformer_blocks.0.attn.to_q`, that it saves
the input **after** AdaLN/norm, and that it saves the corresponding BF16
`to_q` output. Confirm the emitted manifest has both tensor files, dimensions,
dtypes, strides, hashes, checkpoint IDs, and no-FP4-attention setting. A failed
or partial capture must not be treated as a reference.

Run on the RTX PRO 6000, choosing a new run ID (the run directory must not
already exist):

```sh
H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo \
  .venv/bin/modal run baseline/fastvideo_modal.py \
  --run-id projection-exact-20261116 \
  --script projection_capture.py
```

The existing `baseline/fastvideo_modal.py` stores output in the
`h3-megakernel-cache` Modal volume under `/cache/runs/<run-id>/` and records
`driver.log` and `returncode`. If the capture fails, fix it and use a **new**
run ID. Record the successful run link and manifest in a result note.

### 2. Check the exact input in the standalone CUTLASS probe

Extend `megakernel/tests/projection_parity.py` or add a dedicated test to load
the captured `to_q_input.pt` and `to_q_output.pt`. Quantize the captured input
with FastVideo's NVFP4 128×4 scale layout and use the pinned packed `to_q`
weight, scale, and alpha from the checkpoint. Verify the reconstructed
FastVideo `mm_fp4` result against the captured output, then compare the
standalone CUTLASS output against **both**. Explain any numerical tolerance;
do not silently substitute the earlier block-input probe. Run at the full
37,771-row shape, including the final partial tile.

`megakernel/cutlass_probe_modal.py` already builds the extension and executes
the parity test. It will need the new capture run ID or path passed explicitly
to the test. The existing command is:

```sh
H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo \
  .venv/bin/modal run megakernel/cutlass_probe_modal.py --mode verify
```

### 3. Add an all-warp GEMM instruction to Hazy

Adapt the CUTLASS SM120 collective from
`megakernel/src/cutlass_projection.cu` into a Hazy operation. Hazy's 384
threads are split among controller, loader, launcher, storer, and eight
consumer warps. CUTLASS needs all 12 warps at one coordinated task boundary;
calling the standalone `Kernel::operator()` from only the consumer role is
invalid. Add a controller/worker rendezvous with a clear arrival and release
protocol, and ensure normal Hazy instructions still progress. Add a typed
projection parameter record and declare scratch ownership and barriers in
the compiler/resource contract. The 51,200-byte CUTLASS shared storage fits
within Hazy's measured 88,480-byte aligned dynamic region, but its lifetime
must not overlap live Hazy pages.

The standalone grid is 56×296; Hazy has 188 resident CTAs. Implement a
persistent tile assignment so each Hazy CTA can process multiple tiles and
the tail is covered exactly once. A one-tile smoke test is useful, but cannot
pass the full projection gate. Prepare CUTLASS descriptors/parameters on the
host before launch, with no device-side CUDA launches or allocations.

### 4. Pass the one-projection gate and document it

Run the Hazy instruction on the captured input and pinned weight for the
full shape. Compare every BF16 output element with the captured FastVideo
output and report max/mean error, mismatch count, shape, and tolerance.
Repeat launches and test the scheduler's tail and tile coverage; a timed-out
or missing CTA is a failure. Record thread count, shared memory, workspace,
register usage if measured, Modal run link, exact revisions, and command line.
Update the milestone-3 status in `docs/hazy-h3-plan.md` and write a result
note under `results/`. Keep later Q/K/V, VSA, and complete-block work listed
as subsequent gates.

## Relevant files

| Purpose | File |
| --- | --- |
| Revised plan and milestone-3 contract | `docs/hazy-h3-plan.md` |
| Standalone design and measured resources | `docs/cutlass-sm120-projection.md` |
| Existing GPU result | `results/cutlass-projection-sm120-20261116.md` |
| Exact capture draft | `baseline/projection_capture.py` |
| FastVideo GPU harness | `baseline/fastvideo_modal.py` |
| Standalone CUTLASS kernel | `megakernel/src/cutlass_projection.cu` |
| Current parity test and GPU harness | `megakernel/tests/projection_parity.py`, `megakernel/cutlass_probe_modal.py` |
| Hazy dispatch and worker protocol | `megakernel/include/megakernel.cuh`, `megakernel/include/util.cuh`, `megakernel/include/controller/controller.cuh` |
| Graph/compiler contract | `megakernel/compiler.py`, `megakernel/src/ops/graph.cuh` |

Do not describe the standalone CUTLASS pass as Hazy integration. The gate is
complete only after the exact FastVideo capture and full-shape Hazy execution
have both passed on RTX PRO 6000.
