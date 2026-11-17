# H3 on the Hazy megakernel runtime

The pinned reference capture, SM120 runtime smoke, synthetic graph compiler,
Q/K/V/gate projection gates, and first-block Q/K norm, partial RoPE, and VSA
tile gate have passed; see
`results/hazy-reference-sm120-20261115.md`,
`results/hazy-compiler-sm120-20261116.md`, and
`results/hazy-projection-gate-20261116.md`,
`results/hazy-qkv-gate-20261116.md`, and
`results/hazy-qknorm-tile-gate-20261116.md`. The full DiT now runs in one
launch (`results/hazy-dit-gate-20261116.md`); milestone 3 remains in progress
on numerical contract and speed, and milestone 4 has not started.

## Target and boundary

The first complete target is `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer`
at revision `9d0d0d3ae69365aded83fab5562f00f9f06c12aa`, using FastVideo
`ce970eb`, one RTX PRO 6000 Blackwell (SM120), 1344 x 768, 124 frames, eight
denoising steps, and `FASTVIDEO_H3_VSA_FP4=0`. Retain `NVFP4` transformer
linears and `VIDEO_SPARSE_ATTN_H3` at 80% sparsity with 64-token tiles. The
attention arithmetic is BF16: disabling FP4 attention does **not** select dense
attention or change the checkpoint weights. Match FastVideo's trained schedule,
conditioning, latent packing, VSA gate, and decoders. Tokenization and final
media encoding/muxing remain host work. The GPU work from token IDs through
prompt embeddings, generated latents, video frames and audio samples is the
one-launch goal.

The `megakernel/` tree holds the Hazy runtime, the gate operations in
`src/ops/`, and the full-DiT instruction set (`src/h3_dit.cu`,
`src/ops/dit_*.cuh`, `h3_dit.py`). `docs/megakernel-plan.md` describes a different
queue-based design and FP4 attention; this plan uses the per-SM instruction
streams and worker roles in `megakernel/include/`.

## Design for other checkpoints

Separate three things:

1. **Engine:** Hazy controller, loader, launcher, consumer, storer, page and
   semaphore protocol, plus SM120 support. No H3 dimensions or weight formats
   belong in its dispatch code.
2. **Reusable operations:** NVFP4/BF16/FP8 GEMM, dense and block-sparse BF16
   attention, norms, RoPE, activation/gating, quantization, convolution,
   packing, reductions, and latent update. Each operation has an opcode,
   typed parameter record, workspace requirements, synchronization contract,
   and a standalone parity check.
3. **Checkpoint recipe:** a manifest of components (text encoder, DiT, video
   decoder, audio decoder), their weight layouts, block count, precision,
   conditioning mode, VSA policy, denoising schedule, and tensor shapes. A host
   compiler emits per-SM instruction streams, parameters and buffer lifetimes
   from this manifest. No checkpoint-specific branch goes in the persistent
   device loop.

The initial recipe uses 50 blocks and eight steps. A 42-block Trim recipe,
full H3, or an FL2VA recipe can reuse operations and replace or add component
adapters. Changes in architecture, AdaLN representation, attention layout,
conditioning inputs, or decoder still require their corresponding operations
and validation; adding a checkpoint file alone is insufficient. FL2VA must
include first/last-frame preprocessing, image conditioning, and a checkpoint
trained for that mode. The first FastH3 target is T2VA only.

## Milestones and pass conditions

### 0. Freeze the exact reference

- Record checkpoint and FastVideo revisions, configuration, prompt and seed
  from `baseline/prompts.json`, and the no-FP4-attention setting. Never add
  prompts in the harness.
- Capture reference tensors at stage boundaries and selected block boundaries:
  prompt embeddings, packed latents, AdaLN tables, Q/K/V after norm/RoPE,
  VSA masks and indices, sparse-attention output, MLP output, step latents,
  video tiles, audio waveform. Include shape, dtype, stride and numerical
  checksums. Save output video/audio and measured latency/peak VRAM.
- Check which BF16 VSA kernel the reference actually selects. At tile 64,
  FastVideo normally uses its Triton block-sparse path unless the native
  Blackwell path is explicitly enabled; confirm from logs/profile rather
  than treating `FASTVIDEO_H3_VSA_FP4=0` as a kernel name.

**Pass:** one reproducible reference case and a stage-by-stage tensor contract.
The existing FP4-attention capacity numbers are not substituted for this run.

### 1. Make Hazy's scaffold run on SM120

- Add an RTX PRO 6000 build target in `megakernel/Makefile`. Replace the
  H100/B200-only assumptions in `include/config.cuh` and `src/h3_config.cuh`:
  shared-memory budget, 13-page assertion, tensor-memory path, register
  controls, launch bounds and cluster settings. Gate unsupported ThunderKittens
  primitives explicitly and supply SM120 equivalents where needed.
- Reconcile Hazy's 640-thread, 20-warp layout with each imported GEMM and
  attention task's thread/barrier contract. Do not assume a 384-thread kernel
  body can be called unchanged by the consumer warps.
- Run the current mixed-opcode/ring-reuse smoke test on SM120. Extend it with
  page reuse, repeated invocation, and a small cross-SM producer/consumer
  dependency test. Validate physical SM ID indexing and addressable stream
  slots; size the schedule from measured IDs, not grid size alone.

**Pass:** no deadlock or memory error under repeated SM120 smoke runs, with
instrumentation showing all active SMs and correct instruction completion.

### 2. Add a checkpoint-independent compiler and synchronization contract

- Define opcodes and parameter records in `megakernel/src/`; generate typed
  records and per-physical-SM instruction streams from a host-side graph.
  Retain Hazy's controller/worker dispatch and local page protocol.
- Assign tiles and dependency edges explicitly. For cross-SM edges, use global
  completion counters with release/acquire visibility and a proven progress
  rule. A CTA waiting on a producer that cannot become resident can deadlock;
  first use whole-stage waves or another bounded dependency scheme, then
  overlap where the proof and measurements permit it.
- Preallocate activation and scratch buffers from liveness analysis. Keep
  checkpoint weights resident; never allocate or launch CUDA work from inside
  the megakernel. Make each op declare its shared memory, register, barrier
  and global workspace requirements. Reject recipes that exceed the runtime
  limits before launch.

**Pass:** synthetic multi-op graphs run across SMs with correct dependencies,
buffer reuse, and repeatability; instruction streams are inspectable.

### 3. Bring up the DiT with the chosen arithmetic

The first SM120 Hazy `attn.to_q` instruction passed bit-exact parity with the
captured post-RMSNorm/AdaLN FastVideo output at both 128 and 37,771 rows,
including the partial final tile, on three launches each. Q, K, V, and the
active compression-gate projections subsequently passed together in one Hazy
instruction stream at the full shape, again on three launches. They use an
all-warp controller rendezvous and a static persistent CUTLASS tile scheduler
across 188 CTAs; see `results/hazy-projection-gate-20261116.md` and
`results/hazy-qkv-gate-20261116.md`. A Hazy operation now performs Q/K RMSNorm,
96-channel partial RoPE, and four-way padded VSA scatter at full first-block
shape. This operation also passed chained after the four NVFP4 projections
and a cross-SM barrier in one Hazy launch. V/gate are bit-exact and Q/K are
within the stated BF16 tolerance against the frozen reference across three
repeats; see
`results/hazy-qknorm-tile-gate-20261116.md`. A correctness-first Hazy BF16
sparse fine-attention instruction has passed the full first-block shape with
the frozen exact mask, both alone and chained after the projections and Q/K
tile stage in one Hazy launch; see
`results/hazy-sparse-attention-gate-20261116.md`. Dynamic Q/K pooling,
score computation, exact top-k mask construction, V pooling, and the trained
coarse compression branch subsequently passed for step 0 block 0 in a single
Hazy launch using captured Q/K/V/gate inputs; see
`results/hazy-dynamic-vsa-gate-20261116.md`.

The full DiT has since run as one Hazy launch: text refinement, eight steps
of 50 blocks, output heads and latent updates (8,144 instructions, 2,800
NVFP4 GEMMs, 400 tensor-core sparse-attention calls); see
`results/hazy-dit-gate-20261116.md` and `docs/hazy-h3-dit-handoff.md`. Every
new operation passed parity, and every NVFP4 linear and the feed-forward are
bit exact when FastVideo's layer is re-run on the megakernel's intermediate.
Free-running, the launch produces a coherent clip through FastVideo's
decoders, but its final video latents have cosine 0.74 against FastVideo and
it is several times slower. **The milestone pass condition is not met:** the
divergence has not been shown to be within the reference's own sensitivity,
and attention throughput under the 168-register limit of the 384-thread CTA
is open.

- Add SM120 NVFP4 GEMM tasks for Q/K/V, VSA gate, output projection and MLP
  linears. Adapt a CUTLASS-compatible tile scheduler to Hazy instructions;
  preserve the checkpoint's scale and packing format. Validate each task
  against FastVideo's corresponding projection.
- Add BF16 Q/K norm and partial RoPE, tile permutation and inverse,
  pooled-score computation, per-region top-k mask construction, variable
  block sizes, prefix exemptions, dense-step/layer rules, and the trained
  compression-gate branch. These are part of VSA, not optional glue.
- Replace the correctness-first BF16 sparse fine-attention body with a fast
  Hazy device task over the exact tile-64 mask and valid lengths. Use
  vLLM-Omni's block-mask-to-indices contract and FlashInfer's SM120 tile-64
  CuTe kernel as the parity/performance reference. The lower-level FlashInfer
  entry point accepts `block_sizes`, unlike its high-level SM120 wrapper;
  see `results/flashinfer-vllm-vsa-probe-20261116.md`. Its CuTe kernel is
  host-launched with 128 threads, so embedding it in Hazy's 384-thread CTA
  still needs a compatible C++ device implementation or a kernel-body port.
  FP4 Q/K/V quantization and FP4 sparse attention are absent from this recipe.
- Add AdaLN/timestep lookup, residual/gating, norm, SwiGLU, input/output
  projections, modality packing, and the trained eight-step latent update.
  Precompute fixed-schedule AdaLN tables on the host if the reference uses
  them; keep the table format an explicit recipe option.
- Validate in order: one projection, one attention call with exact mask,
  one complete block, 50 blocks for one step, then eight steps. At each gate,
  compare internal tensors and the produced video/audio latents to the
  frozen reference, then time it. Do not use end-video similarity to hide a
  block-level mismatch.

**Pass:** full DiT latents pass the numerical contract for the fixed reference
case and no attention FP4 path is invoked.

### 4. Add the remaining GPU stages

- Text encoder: adapt the checkpoint's trimmed Qwen3-VL components, NVFP4
  linears, BF16 attention and normalization into the op library. The host
  still tokenizes; the encoder writes the same embeddings as FastVideo.
- Video decoder: port the lightweight 26-block INT8/ConvRot VAE, its attention,
  convolution/layout operations, 28-tile decode and tile stitching. ConvRot
  here is a **decoder** implementation detail; it does not require the
  Comfy-Org transformer or workflow.
- Audio decoder: port the audio VAE's convolution and output operations.
  Small runtime share does not remove it from the one-launch completion gate.
- Compile the full graph into one launch. Host code may copy inputs, prepare
  descriptors and encode/mux final media; it must not call FastVideo GPU
  stages between launch and output readback.

**Pass:** the one-launch GPU pipeline yields reference-matching video frames
and audio samples for the same prompt/seed/shape, and encodes a playable clip.

### 5. Measure and extend

- Report cold and warm latency, GPU time, peak VRAM, launch count and per-stage
  time against the exact no-FP4-attention FastVideo recipe at 832 x 480 and
  1344 x 768. Compare quality only for identical prompts/seeds/configuration.
- Add a second checkpoint as an architecture test of the component interface:
  a different precision or block-count recipe must produce a new instruction
  graph without edits to the engine. Then add components needed by higher
  quality or FL2VA checkpoints, validating each against its own reference.

**Pass:** the first recipe works end to end, measured gains and regressions
are visible, and the second recipe demonstrates reuse rather than a copied
monolithic kernel.

## Principal risks

- **SM120 is a real port.** The copied Hazy runtime builds only for H100/B200
  today; a compiler flag change alone cannot establish correctness.
- **BF16 VSA is still difficult.** The chosen quality path avoids FP4 attention
  arithmetic but retains dynamic block selection and sparse attention.
- **Thread/resource mismatch.** Hazy's worker roles and shared pages may not
  accommodate a library kernel's full CTA without adaptation.
- **One-launch dependency safety.** Cross-SM waits require a residency and
  memory-ordering argument, plus a watchdog test for deadlock.
- **Capacity.** The existing 68 GiB peak is for FP4 attention. Re-measure
  the selected BF16 VSA path and budget instructions, buffers and workspaces.
