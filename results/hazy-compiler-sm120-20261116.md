# Hazy graph compiler bring-up on RTX PRO 6000

Milestone 2 in `docs/hazy-h3-plan.md` was validated on 2026-11-16 for the
FastVideo no-FP4-attention target. The compiler is checkpoint-independent;
its GPU opcodes in this milestone are synthetic affine/add operations, not
FastVideo DiT projections or attention.

- GPU: Modal RTX PRO 6000 Blackwell (SM120), 188 measured unique physical SM IDs.
- Run: https://modal.com/apps/r19morris/main/ap-5aZh1zC7Th4qiwoNft8Atr
- `make GPU=RTXPRO6000 smoke`: passed mixed opcodes, two-slot ring reuse,
  cross-SM rendezvous, and repeat launch.
- `make GPU=RTXPRO6000 graph`: passed five whole-stage waves, explicit DAG
  dependencies, branch fanout, disjoint SM tiles, activation arena reuse,
  release/acquire barriers, exact numerical output, and repeat launch.
- Local host checks passed for generated records, physical-ID addressing,
  liveness reuse, bad dependency and resource rejection, Python syntax, and
  `git diff --check`.

The scheduling proof is limited to one CTA per measured SM: the 188-CTA smoke
test established residency for this runtime configuration. The compiler emits
one node per whole-stage wave and a global barrier between nodes. It currently
rejects operations that require shared memory beyond the runtime's existing
100,000-byte allocation. Real NVFP4 GEMM and BF16 VSA operations need their
own resource contracts and parity validation in milestone 3.
