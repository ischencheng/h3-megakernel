# Understanding the H3 megakernel

A short guide to what the megakernel is, how the pieces fit, and how to
rebuild it by hand. For status and numbers see
`results/hazy-dit-gate-20261116.md`; for open work see
`docs/hazy-h3-dit-handoff.md`.

## The idea in one paragraph

Normally PyTorch launches thousands of small GPU kernels, one per operation.
A megakernel launches **one** kernel that stays on the GPU and runs a list of
instructions the host wrote beforehand, like a tiny virtual machine. The host
decides *what* to run and in what order; the GPU only executes. Here that one
launch covers the whole FastH3 diffusion transformer (DiT): 8 denoising
steps of 50 blocks.

## The four pieces

| Piece | Where | What it does |
| --- | --- | --- |
| Runtime | `megakernel/include/` | The virtual machine, copied from Hazy Research. Fetches the next instruction and dispatches on its opcode. |
| Operations | `megakernel/src/ops/` | The GPU code for each opcode: GEMM, norm, quantize, attention, and so on. |
| Kernel binding | `megakernel/src/h3_dit.cu` | Lists which operations this kernel contains and exposes `run` to Python. |
| Host compiler | `megakernel/h3_dit.py` | Builds the instruction list and the buffer table, then launches once. |

## What actually gets sent to the GPU

Two things:

1. **An instruction list.** Each instruction is 32 integers: an opcode, then
   arguments (buffer indexes, row and column counts, flags, a few scalars).
   Every SM (the GPU's compute units, 188 on this card) runs the same list
   in order.
2. **A buffer table.** A list of GPU memory addresses. Instructions refer to
   their inputs and outputs by index into this table.

The kernel cannot allocate memory, so every buffer exists before launch.

In `h3_dit.py`, `p` is a `Program` object. Calls like `p.linear(...)` or
`p.emit("VSA_POOL", ...)` do not run anything; each appends one instruction
to the list. `p.run()` sends the list and launches.

## How one instruction executes

- One block of 384 threads (12 warps of 32) sits on each SM.
- For every DiT operation, all 12 warps do the work together. The work is
  split by striding: block `b`, warp `w` takes items
  `b*12 + w, + 188*12, ...`, so every item is done exactly once across the GPU.
- After the operation, each SM increments a shared counter and waits until
  all 188 have arrived. This **wave barrier** guarantees a stage is finished
  everywhere before the next one reads its output.

## What one transformer block looks like

About 20 instructions, in this order (`h3_dit.py` lines 573-630):

1. `RMSNORM_MOD` - normalize and apply the timestep modulation.
2. `NVFP4_QUANT` - pack activations to 4 bits.
3. `PROJECTION` x4 - Q, K, V and gate matrix multiplies.
4. `QKNORM_TILE` - normalize Q/K, apply rotary positions, arrange tokens
   into 64-token tiles.
5. `VSA_POOL`, `VSA_SCORES`, `VSA_TOPK` - decide which tiles each query tile
   attends to (keeps 20% of video tiles).
6. `SPARSE_ATTENTION_MMA`, `VSA_COMPRESS` - attention over the kept tiles.
7. `NVFP4_QUANT`, `PROJECTION` - output projection.
8. `RMSNORM_MOD` - add the attention result, normalize again.
9. `NVFP4_QUANT`, `PROJECTION`, `NVFP4_QUANT`, `PROJECTION` - the
   feed-forward network.

Around the 50 blocks, each step also computes the timestep embedding, packs
latents and text into one sequence, and applies the latent update.

## What is and is not in the megakernel

Inside one launch: the whole DiT. Outside it, still run by FastVideo: the
text encoder, the video decoder and the audio decoder. Tokenizing text and
encoding the MP4 stay on the CPU by design.

## Restarting from scratch: suggested order

Build the smallest thing that can fail visibly, then grow it.

1. **Runtime smoke test.** One opcode that writes its SM ID. Confirms the
   kernel launches, every SM runs, and the barrier works. Run it twice.
2. **One trivial operation end to end.** Elementwise add over a buffer
   table. This forces you to write the pointer table, the striding, the
   instruction layout and the Python builder while the math is trivial.
3. **Row-wise operations.** RMSNorm, then the quantizer. Simple loops, easy
   to compare exactly.
4. **One GEMM.** Hardest to integrate (it needs all 12 warps and the shared
   memory). Reuse a library kernel (CUTLASS) rather than writing your own.
5. **Attention, slow first.** Write a plain one-warp-per-query version that
   is obviously correct. Only then write the tensor-core version and test it
   against the slow one.
6. **One block, then 50, then 8 steps.** Add nothing new at these stages;
   they only chain operations already tested.
7. **Optimize last,** and only what a profile points to.

## Writing an operation by hand

Each operation needs four things:

- **An opcode** in `src/h3.cuh`.
- **A body**: a function all 384 threads run. Read arguments from the
  instruction, fetch buffer pointers from the table, loop over your share of
  the work. `dit_common.cuh` wraps a body into a full operation.
- **A documented argument layout**: which of the 32 words means what. Keep
  the comment next to the body; the Python side must match it exactly.
- **A line in the kernel binding** so the kernel includes it.

Keep bodies free of model-specific constants. Sizes come from the
instruction, data from the buffer table.

## Testing for correctness

Use three levels. Each catches what the others miss.

1. **Operation parity on synthetic data** (`tests/dit_ops.py`). Random
   inputs, one instruction, compare with the real reference function. Include
   awkward shapes: a partial last tile, a row count that is not a multiple of
   128, a zero block, very large and very small values.
2. **Teacher forcing on real data.** Take the megakernel's own intermediate,
   feed it to the reference layer, compare outputs. This isolates one segment
   and removes accumulated drift. It is how the linears were shown bit exact.
3. **Free-running comparison.** Run everything from the same start and
   compare final outputs. Useful as a sanity check, weak as proof.

Decide in advance what "equal" means for each operation:

- Integer or selection results (quantized codes, masks): **bit exact**.
- Floating-point reductions: a stated tolerance, and report the count of
  elements outside it, not just the maximum.

Other habits that paid off:

- Verify the reference is deterministic before trusting it.
- Report mismatch counts, max error, relative error and cosine together.
- Always check completion, not only values: every SM reached every barrier.
- Keep a slow, obviously-correct version of anything clever.

## Pitfalls met in this project

- **Quantization amplifies noise.** 4-bit activations turn rounding-level
  differences into several percent per block. Whole-block outputs will not
  match elementwise even when every piece is exact. Plan the test strategy
  around this from day one.
- **Register limit.** 384 threads per block leaves 168 registers per thread.
  Code that needs more silently spills to memory and runs several times
  slower. Check with `cuobjdump -res-usage`.
- **Compiler flag.** Build with `-gencode=arch=compute_120a,code=sm_120a`;
  the `-arch=sm_120a` shorthand rejects the 4-bit conversion instruction.
- **Memory operations are expensive here.** Lay data out so one wide load
  feeds several tensor-core instructions.
- **No local GPU.** Every test is a remote run of a few minutes. Batch
  checks into one run and use `modal run --detach` for long ones.

## Glossary

- **SM**: streaming multiprocessor, one of the GPU's compute units.
- **CTA**: one block of threads; here one per SM.
- **Warp**: 32 threads that execute together.
- **NVFP4**: NVIDIA's 4-bit floating-point format with per-block scales.
- **GEMM**: general matrix multiply.
- **VSA**: video sparse attention; attends to a selected subset of tiles.
- **AdaLN**: normalization whose scale and shift depend on the timestep.
- **Wave barrier**: the wait that keeps all SMs on the same stage.
