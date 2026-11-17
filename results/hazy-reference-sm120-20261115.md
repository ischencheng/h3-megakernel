# Hazy SM120 and BF16 VSA reference verification

The first two implementation items in `docs/hazy-h3-plan.md` were verified on
2026-11-15 on a Modal RTX PRO 6000 Blackwell (SM120).

## Pinned reference

- Checkpoint: `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer` at
  `9d0d0d3ae69365aded83fab5562f00f9f06c12aa`.
- FastVideo source checkout: `ce970eb715215c79a169658d7aa6c8a96e64891e`.
- `FASTVIDEO_H3_VSA_FP4=0`, `FASTVIDEO_VSA_PTX_BLACKWELL=0`, NVFP4 DiT
  linears, `VIDEO_SPARSE_ATTN_H3`, 80% sparsity, 64-token tiles.
- Prompt: index 0 in `baseline/prompts.json` (basketball during recess), SHA256
  `8a0d0914189356d56447b166cd102d0fe72b131d5ee1bea04baa58e9df218f70`.
  Seed 1101, 1344 x 768, 124 frames, eight transformer forwards.
- Modal run: https://modal.com/apps/r19morris/main/ap-7yWwpNLhOWtq2J9cMeryfQ
- Volume: `h3-megakernel-cache/runs/reference-bf16-vsa-20261115/`.

`baseline/reference_capture.py` saved 43 tensors, including prompt embeddings,
initial/final video and audio latents, decoded video and audio, and inputs,
Q/K/V, gate, valid lengths, masks, attention outputs and block outputs for
selected first/last blocks and steps. The manifest has individual SHA256
hashes, shapes, dtypes and strides. Its `verified` flag is true. Every one of
the eight steps recorded 50 transformer-block calls, 50 VSA calls and 50 mask
constructions. An intercepted FP4-attention call would have failed the run.

The saved MP4 has a video track (`avc1`) and an audio track (`mp4a`). The first
request took 273.8 seconds end to end, including first-use JIT and VAE
compilation plus tensor capture; this is not a warm latency benchmark.
FastVideo logged a maximum stage peak allocation of 67,480 MiB during
denoising. Stage logging resets PyTorch's peak counter, so the manifest's final
`peak_allocated_bytes` field is not the whole-request peak.

## SM120 runtime smoke

`make GPU=RTXPRO6000 smoke` compiled and passed twice after the final code
change. The test used 188 CTA blocks with unique physical SM IDs, mixed
opcodes, repeated instruction-ring slots, page release, a bounded cross-SM
rendezvous with published global data, and two consecutive launches.

- Final Modal run: https://modal.com/apps/r19morris/main/ap-eg4hS93EDuahAeF2zHjOCb
- SM120 runtime: 384 threads, eight consumer warps, five 16-KiB pages,
  100,000-byte total shared-memory budget.
- The first compiler attempt rejected `setmaxnreg.inc/dec` for SM120. The
  final build omits those instructions on SM120 and passes the smoke test.
- The SM100 tensor-memory path is disabled for this build. No H3 tensor
  operation has yet been ported or validated under the new worker layout.

Local Python syntax checks and `git diff --check` also passed. H100 and B200
builds were not rerun at the time of this reference capture.

## SM100 and SM120 build switching

The build now writes `mk_h3` to `megakernel/build/<GPU>/`, and the smoke
command imports from that directory. `B200` uses explicit
`compute_100a -> sm_100a` code generation; the previous `-arch=sm_100a`
shorthand lowered architecture-specific inline PTX to plain SM100 and failed
on `tcgen05` and `setmaxnreg`. On 2026-11-15, a single Modal container built
both architecture outputs and reran SM120 smoke without `make clean`:
https://modal.com/apps/r19morris/main/ap-l14W0wdoPcOOzSh3dziFWi

The corrected B200 target also compiled and passed the smoke test on a B200:
148 unique physical SM IDs, mixed opcodes, ring reuse, cross-SM rendezvous,
and two consecutive launches:
https://modal.com/apps/r19morris/main/ap-mbYcRTL9VcwhvqH272Zysb

H100 was not rerun. The smoke result proves the runtime scaffold on both
architectures; it does not prove H3 operation portability between them.
