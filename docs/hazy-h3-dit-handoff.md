# Handoff: full FastH3 DiT in one Hazy launch (RTX PRO 6000, SM120)

Written 2026-11-16, updated after the full gate. Continues `docs/hazy-h3-plan.md` milestone 3 from the
state in `results/hazy-dynamic-vsa-gate-20261116.md`. Nothing below is
committed yet; inspect `git status --short` first.

## What exists now

| Piece | File | State |
| --- | --- | --- |
| All-warp op wrapper, pointer-table buffers, folded cross-SM wave barrier | `megakernel/src/ops/dit_common.cuh` | GPU-validated through the op tests |
| RMSNorm + AdaLN + gated residual, NVFP4 activation quantize (with untile gather and fused SwiGLU source), generic dense linear, timestep embedding / SiLU / flow update, row copy | `megakernel/src/ops/dit_ops.cuh` | **Passed** op parity |
| Q/K norm + RoPE + tile scatter, pool, scores, top-k, warp-per-query attention, **tensor-core (mma) sparse attention**, coarse compression — all parameterized by the instruction | `megakernel/src/ops/dit_attention.cuh` | **Passed** op parity |
| Full-DiT kernel binding `mk_h3_dit` (`run`, `prepare`, `params_words`, `opcodes`) | `megakernel/src/h3_dit.cu`, `make GPU=RTXPRO6000 hazy_dit` | Builds on SM120 |
| Shared CUTLASS host preparation (factored out of `h3_projection.cu`) | `megakernel/src/ops/projection_host.cuh` | Builds in both modules; old gate re-passed |
| Host compiler: `Program`, `DiTRecipe`, `compile_dit`, `recipe_from_fastvideo` | `megakernel/h3_dit.py` | Used by both tests |
| Op parity test | `megakernel/tests/dit_ops.py` | **Passed**, see below |
| Block / 50-block / 8-step gates + decode from megakernel latents | `megakernel/tests/hazy_dit.py` | Block, step and full gates run; see Results |
| Attention stage benchmark and binary inspection | `megakernel/tests/dit_attention_bench.py`, `dit_binary_inspect.py` | Used for the profile below |
| Model fact probe | `baseline/dit_introspect.py` | Ran; output in volume `runs/dit-introspect-20261116-a/introspection.json` |

The DiT target must be built with
`-gencode=arch=compute_120a,code=sm_120a`; the `-arch=sm_120a` shorthand
makes ptxas reject `cvt.rn.satfinite.e2m1x2.f32`.

## Verified results so far

Op parity run (all passed):
https://modal.com/apps/r19morris/main/ap-sVlC3UrFlFME1mfr0Hr3rd

- NVFP4 quantize: codes and 128x4 scales **bit exact** against flashinfer
  for global scales 1.0, 2.33 and 0.034, with row gather, and with the
  SwiGLU source.
- RMSNorm/AdaLN variants, dense linear, SwiGLU, Q/K norm: zero elements
  outside `atol=0.03125, rtol=0.015625`; residual hidden states, timestep
  embedding, SiLU and flow update bit exact.
- Tensor-core fine attention vs FastVideo's Triton kernel (56 heads, 40
  tiles, partial tiles): max error 0.0039, 14,404 of 18,350,080 elements
  differ. The warp-per-query op: max 0.0156, 5.6M differ. Chain time 6.5 ms
  vs 28.1 ms at that size.

Model facts the recipe relies on (from the probe): 37,771 rows = 61 text,
414 audio, 37,296 video; 668 tiles, 8 prefix, keep 132; two row timesteps per
step (video+text, audio); Q/K/V/gate/to_out use unit activation scale,
`fc_in`/`fc_out` carry per-layer `_nvfp4_input_global_sf`; all 50 gates
active; AdaLN, refiner and `context_embedder` are dense BF16, in/out heads
and time MLP FP32; refiner uses SDPA with no fusions; scheduler is
`MiniMaxH3Scheduler` (non-stochastic).

## Results

Full write-up: `results/hazy-dit-gate-20261116.md`. Artifacts:
`results/hazy-dit-20261116/` (clip, frame, `report.json` from run `-d`).

- The whole DiT ran in **one launch** (8,144 instructions). FastVideo's
  decoders turned its latents into a coherent clip of the prompted scene.
- **Every NVFP4 linear and the feed-forward are bit exact** when FastVideo's
  layer is re-run on the megakernel's intermediate. FastVideo itself is
  deterministic against the frozen reference.
- Free-running it is **not a reproduction**: block 0 output 2.2% relative
  L2, final video latents cosine 0.742, decoded PSNR 11.8 dB. Four-bit
  activation quantization amplifies BF16-rounding-level differences
  (0.37% in, 5.8% out across the feed-forward). Whether that alone explains
  0.74 is **not established**.
- It is slow: 526 s for eight steps in run `-d` against FastVideo's 51 s.
  Attention was the whole gap (1,240 ms of a 1,300 ms block).
- Cause: the 384-thread CTA caps threads at 168 registers
  (`REG:168 STACK:1152`), so attention accumulators spill around every
  `mma`. The op now stages K/V tiles in shared memory in fragment order and
  measures **459 ms** per call on the full-shape synthetic benchmark; that
  projects to about 210 s for eight steps.

## Runs

| Run id | What | State |
| --- | --- | --- |
| `hazy-dit-20261116-a` | first block/step; 8-step failed on a full parameter table (fixed) | superseded |
| `-b` | block diagnostics; OOM in diagnostics (fixed) | superseded |
| `-c` | block gate with teacher-forced checks and profile | complete |
| `-d` | block + step + full + decode, first tensor-core attention | complete, the numbers of record |
| `-e` | same with 459 ms attention | **cancelled by a client disconnect before denoising; directory exists, no results** |
| `-f` | rerun of `-e`, launched with `modal run --detach` | **also cancelled from Modal's side, 13 s into FastVideo's own denoising; no results** |

All ids above are taken; run directories must not already exist.

## Next steps

1. Get an end-to-end timing with the 459 ms attention. Runs `-e` and `-f`
   both received an external cancellation signal with no reason in the app
   log (`modal app logs ap-03Ck53nfqSxbrEXc8mIUEU`), at different stages,
   before any megakernel code ran. Find out why (workspace spend limit,
   preemption, or a manual stop) before spending on a third attempt. The
   459 ms figure is from the synthetic benchmark only. Then rerun with a
   fresh id:

   ```sh
   H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo .venv/bin/modal run --detach \
     megakernel/cutlass_probe_modal.py --mode dit --script hazy_dit.py \
     --args "--out /cache/runs/<fresh-run-id> --gates block,full --pipeline megakernel"
   ```

   `--gates block --pipeline reference` is the quick form; `--slow-attention`
   swaps in the warp-per-query op. Then replace the timing rows in the
   result note and the clip in `results/hazy-dit-20261116/`.
2. Decide the numerical contract. Measure FastVideo's own sensitivity: feed
   its denoising loop a block-0 input perturbed by one BF16 ulp on a small
   fraction of elements and compare final latents. If its spread is far
   below cosine 0.74, there is a real defect to find; start with the 0.76%
   refined-text difference (dense BF16 refiner, FP32 accumulation here vs
   eager BF16 ops there) and then compare block outputs block by block
   within step 0 to find where error growth departs from the per-block rate.
3. Attention speed. Benchmark with
   `--mode dit-ops --script dit_attention_bench.py` (stage mask in
   instruction word 9; never set in a recipe) and inspect spills with
   `--script dit_binary_inspect.py`. Untried: two passes over half the head
   channels to halve the accumulators, accumulators in shared memory, or a
   CTA shape with fewer threads for the attention stage.
4. Other per-block costs once attention is fixed: compress 14 ms, scores
   8 ms, Q/K norm 7 ms; `proj_in` 67 ms per step.

## Still open after the DiT (milestone 4)

Text encoder (trimmed Qwen3-VL), INT8/ConvRot video decoder with 28-tile
stitching, and the audio VAE are **not ported**. `hazy_dit.py` runs them as
FastVideo stages around the one-launch DiT. Until they are instructions in
the same stream, the pipeline is not one launch end to end.

## Known limits of the current recipe

`recipe_from_fastvideo` rejects anything outside the pinned T2VA request:
conditioning rows, capacity padding, dense layers / dense first steps,
token reduction, compete mode, more than one sparse video span, rank-reduced
AdaLN, PDD heads, calibration-table or dynamic activation scales, and more
than one text tile (64 tokens). Head width is fixed at 128 channels in the
attention ops.
