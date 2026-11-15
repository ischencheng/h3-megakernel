# H3 megakernel scaffold

The same execution engine as HazyResearch/Megakernels, with an H3 entry point
and a small dispatch smoke operation. **No H3 inference operations are implemented.**
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

## Reference build (Linux + CUDA, H100/B200 only)

Install PyTorch and pybind11 in the active Python environment, then:

```sh
cd megakernel
make bootstrap
make GPU=H100           # or GPU=B200, with a matching CUDA toolkit/GPU
make GPU=H100 smoke
```

Set NVCC, PYTHON and THUNDERKITTENS_ROOT if necessary. PYTHON defaults to
python3 and requires the corresponding python3-config executable.
Switching GPU targets requires `make clean` before rebuilding.
The dependency checkout is ignored by Git; its revision is pinned in bootstrap.py.

## RTX PRO 6000 / SM120 port required

The existing H3 project targets SM120. This scaffold deliberately preserves the
Hazy H100/B200 runtime; it is NOT a working SM120 backend. Do not compile it by
simply substituting sm_120a for sm_100a. Upstream assumes 13 x 16 KiB shared-memory
pages, architecture-specific register controls, and (on its Blackwell path)
tensor-memory allocation/operations. These require an SM120-compatible runtime
configuration and primitives, then validation of residency and synchronization.
The current 640-thread warp-role scheme must also be reconciled with any imported
CUTLASS task's thread participation and barrier assumptions.

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

Python syntax, build-command generation and copied-source checks can run locally.
CUDA compilation and GPU smoke tests require the reference GPU environment above.
