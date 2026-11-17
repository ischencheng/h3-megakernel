# H3 megakernel scaffold

The same execution engine as HazyResearch/Megakernels, with an H3 entry point,
dispatch smoke operations, a checkpoint-independent synthetic graph compiler,
and the first FastH3 NVFP4 Q/K/V and compression-gate projections as all-warp
instructions in one stream. The first-block BF16 Q/K norm, partial RoPE, and
four-way VSA tile scatter, followed by sparse BF16 fine attention with the
frozen mask, also pass their numerical gates in one Hazy launch on SM120.
The remaining DiT and decoder operations are still open.
See UPSTREAM.md for pinned source revisions and licensing.

## Layout

- `include/megakernel.cuh`: entry point, shared state setup, warp roles.
- `include/config.cuh`: upstream resource configuration.
- `include/util.cuh`: state, opcode dispatch, worker-loop macro.
- `include/{consumer,loader,storer,launcher,noop}.cuh`: worker roles and NoOp.
- `include/controller/`: instruction fetch, page allocation, semaphore setup, timings.
- `src/h3_config.cuh`: local configuration overrides.
- `src/h3.cuh`: H3 globals and opcode definitions (model buffers to be added).
- `src/ops/smoke.cuh`: minimal operation implementing the existing interface.
- `src/h3.cu`: operation registration and Python binding.
- `tests/smoke.py`: mixed-opcode dispatch and instruction-ring reuse check.
- `compiler.py`: typed synthetic graph, SM assignment, wave schedule, resource
  validation, and activation arena lifetime planning.
- `src/ops/graph.cuh`: synthetic affine/add tasks and release/acquire wave barrier.
- `tests/graph.py`: multi-op dependency, arena reuse, and repeatability check.
- `src/ops/projection.cuh`, `src/h3_projection.cu`: SM120 CUTLASS projection
  instruction, controller rendezvous, host-prepared parameters and binding.
- `tests/hazy_projection.py`, `tests/hazy_qkv.py`: exact-input and repeated
  full-shape parity for one projection and four projections in one launch.
- `src/ops/qknorm_rope.cuh`, `tests/hazy_qknorm_rope.py`: all-warp BF16 Q/K
  norm, partial RoPE, and padded VSA tile scatter, checked against the frozen
  first-block attention inputs.
- `tests/hazy_projection_attention.py`: four NVFP4 projections, a cross-SM
  barrier, the BF16 VSA boundary, another barrier, and sparse fine attention
  in one Hazy instruction stream.
- `src/ops/sparse_attention.cuh`, `tests/hazy_sparse_attention.py`: rewritten
  BF16 block-sparse attention and full first-call comparison to FastVideo's
  test-only Triton oracle.

## Reference build (Linux + CUDA)

Install PyTorch and pybind11 in the active Python environment, then:

```sh
cd megakernel
make bootstrap
make GPU=B200 smoke       # SM100
make GPU=RTXPRO6000 smoke # SM120
make GPU=RTXPRO6000 graph # SM120 synthetic compiler and GPU contract
python3 scripts/bootstrap_cutlass.py
make GPU=RTXPRO6000 cutlass_projection # standalone NVFP4 projection probe
make GPU=RTXPRO6000 hazy_projection    # all-warp Hazy projection binding
```

Set NVCC, PYTHON and THUNDERKITTENS_ROOT if necessary. PYTHON defaults to
python3 and requires the corresponding python3-config executable.
Each target builds in `build/<GPU>/`, so switching targets does not reuse the
other architecture's extension or require `make clean`.
The dependency checkout is ignored by Git; its revision is pinned in bootstrap.py.
CUTLASS is separately pinned by `scripts/bootstrap_cutlass.py`.

## RTX PRO 6000 / SM120 smoke port

`GPU=RTXPRO6000` selects a 384-thread smoke configuration with five 16-KiB
pages under a 100,000-byte shared-memory budget. It avoids the upstream SM100
tensor-memory allocator. The smoke test checks physical SM IDs, mixed opcodes,
page release, ring reuse, a bounded cross-SM rendezvous and repeat launches.
The first H3 tensor operation uses this layout; later operations need their
own thread and barrier contracts.

## Implementing H3 operations

1. Port and validate the runtime on the target GPU first.
2. Add weight/activation views and dependency counters to `h3::globals`.
3. Assign unique opcodes; zero is reserved for upstream NoOp.
4. Implement controller init_semaphores/release_lid and loader, launcher,
   consumer, storer run hooks. Replace the smoke NoOp page protocol when using pages.
5. Register each operation type in the template arguments in src/h3.cu.
6. Generate instruction streams (opcode in integer 0; remaining arguments are
   operation-specific). Current upstream selects streams by physical SM ID,
   not blockIdx.x. Production scheduling must account for that explicitly.
7. Validate operations against the existing H3 reference tensors before fusion.

The earlier docs/megakernel-plan.md describes a different queue-based design;
this directory is the requested Hazy-style starting point and does not implement
that plan's global queue. Existing model stages are untouched.

## Validation status

On 2026-11-15, `GPU=RTXPRO6000` built and passed `tests/smoke.py` on a Modal
RTX PRO 6000: 188 unique physical SM IDs, mixed opcodes, two-slot ring reuse,
a cross-SM rendezvous, and two consecutive launches. `GPU=B200` also built
and passed on a Modal B200 with 148 unique physical SM IDs. Both architecture
outputs were built in one container without overwriting each other. The H100
build was not rerun. The first H3 projection subsequently passed; see below.

## Compiler bring-up on SM120

`compiler.py` accepts a topologically ordered graph with explicit input names,
output names, tensor lengths, and operation resource declarations. It assigns
one disjoint tile per measured physical SM ID for each node and emits a task
instruction followed by a global barrier instruction. All 188 CTAs must be
resident; the host requires exactly one unique measured ID per CTA. The barrier
publishes each wave with device-scope release/acquire ordering. No CUDA work is
allocated or launched inside the persistent kernel.

The compiler uses aligned arena offsets and last-use analysis. It rejects bad
dependencies, unknown ops, incompatible tensor lengths, unaddressable physical
IDs, oversized arena requests, and resource declarations beyond the current
runtime. The synthetic graph compiler rejects extra shared-memory requests;
the projection compiler declares exclusive ownership of the dynamic page
region in a separate resource contract. The generated
`Compiled.describe()` data exposes buffers and waves for inspection.

The synthetic graph test runs five affine/add waves with a branch and arena
reuse, then repeats the launch and compares every output element. The same
mechanism will schedule further DiT ops. See
`results/hazy-compiler-sm120-20261116.md`.

## SM120 CUTLASS projection probe

`src/cutlass_projection.cu` adapts NVIDIA CUTLASS example 79a to the FastH3
DiT `attn.to_q` weight shape and BF16 output. `tests/projection_parity.py`
uses the packed checkpoint weight and FastVideo's activation quantizer to
compare one projection with the FastVideo FP4 matmul. It also reports CTA
threads, shared memory, grid shape, and workspace. The standalone feasibility
probe passed on an RTX PRO 6000
at 128 and 37,771 rows with exact BF16 output parity, 384 threads/CTA, 51,200
shared bytes, and zero workspace. The subsequent exact capture and Hazy
instruction passed at the full shape with zero BF16 mismatches on three
launches. Q/K/V and the active gate then passed together in one Hazy launch;
see `results/hazy-projection-gate-20261116.md` and
`results/hazy-qkv-gate-20261116.md`.
