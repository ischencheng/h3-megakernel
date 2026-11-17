# The five Hazy roles and how the DiT could use them

Companion to `docs/megakernel-guide.md`. That guide describes the DiT as it
is built today, where every operation is one function run by all warps. This
note explains the role design the Hazy runtime was built around, and what
using it properly would look like here.

Scope note: the role mechanics below come from the runtime headers in
`megakernel/include/`. This repository does not contain Hazy's own model
operations, so the description of how upstream uses each role is the
intended design, not code you can read here.

## The five roles

Each block of threads on an SM is split into warps with fixed jobs. On this
GPU (SM120 config): 12 warps, 5 shared-memory pages of 16 KiB, and a
two-slot instruction pipeline.

| Role | Warps | Job |
| --- | ---: | --- |
| Controller | 1 | Fetches the next instruction, assigns shared-memory pages to it, sets up its semaphores, announces it. |
| Loader | 1 | Copies the instruction's inputs from global memory into pages. |
| Launcher | 1 | Issues tensor-core work when the hardware separates issue from compute. A no-op otherwise. |
| Consumer | 8 | Do the arithmetic on data sitting in pages. |
| Storer | 1 | Writes results to global memory and signals completion. |

An operation is a struct with one `run` function per role, plus two
controller hooks: `release_lid` (which page this instruction frees first)
and `init_semaphores`.

## How they cooperate

- **Pages** are the shared scratch. The controller hands an instruction its
  pages in the order the previous instruction released them. A role calls
  `wait_page_ready` before touching a page and `finish_page` when done.
- **Semaphores** are the hand-offs inside one instruction: loader to
  consumers ("tile is loaded"), consumers to storer ("result is ready").
  Each instruction slot has 32 of them.
- **Two instruction slots** are live at once. The controller prepares
  instruction N+1 while N is still computing, and the loader can start
  filling pages for N+1 as soon as N releases them. This overlap is the point
  of the design: loading, computing and storing run concurrently instead of
  in sequence.
- **Cross-SM dependencies** are global counters the storer increments and a
  later instruction waits on.

## What this project does today

Every DiT operation is wrapped by `all_warp_op` in
`megakernel/src/ops/dit_common.cuh`. All five roles get the same `run`: do
the default page release, meet at a block-wide barrier, execute one shared
body, meet again, then wait at a GPU-wide wave barrier.

Why: the CUTLASS GEMM needs all 12 warps and all the shared memory, so it
could not live in the consumer role alone. Making every operation all-warp
kept one simple contract.

What it costs:

- No overlap. Each instruction fully finishes before the next starts.
- Every stage ends with a wait for all 188 SMs.
- Compute warps also do the loading. In the attention this is measurable:
  on the full-shape benchmark, staging tiles is about 85 ms of a 459 ms call,
  and it sits in the same warps as the tensor-core work.

## A role-split DiT

What each role would do for the main operation families:

| Operation | Loader | Consumers | Storer |
| --- | --- | --- | --- |
| Row-wise (norm, quantize, SwiGLU, residual) | Bring a chunk of rows into pages | Reduce and transform in shared memory | Write rows back |
| Pool, scores, top-k | Bring pooled rows or a score row | Reduce / select | Write pooled values or mask |
| Sparse attention | Stage the next K and V tile into a spare page while the current one is in use | Tensor-core QK, softmax, PV on staged tiles | Write the 64 finished query rows |
| GEMM | See the next section | | |

The attention gains most. Today a group stages K, computes, stages V,
computes, with barriers between. With a loader warp and double buffering,
the next tile is already in a page when the consumers finish the current
one, and the consumers run nothing but tensor-core and softmax code.

Dependencies would move from "all SMs finish the stage" to counters per
tile or per row range, so an SM can start the next operation on data that is
already complete.

## Third-party code

- **CUTLASS GEMM.** It already has the same split internally: a load side,
  a tensor-core mainloop and an epilogue. Two ways to use it:
  1. Keep it as an exclusive all-warp instruction, as now. Zero porting; it
     is already at full speed (about 11 ms for the largest GEMM). Role-split
     and all-warp instructions can coexist in one stream because the wrapper
     follows the page protocol.
  2. Map its collectives onto roles. Large effort, and the only gain is
     overlapping a GEMM with its neighbours.

  Recommendation: option 1. The GEMMs are not the bottleneck.
- **ThunderKittens.** Provides shared-memory tile types, semaphores and
  bulk-copy helpers the runtime already uses. Its Hopper and SM100 paths do
  not all apply to SM120; each primitive needs checking before use.
- **FlashInfer / FastVideo attention kernels.** They are host-launched
  kernels with their own thread count and cannot be called from inside the
  megakernel. Use them as the parity and performance reference, not as code
  to embed.
- **flashinfer quantizer.** Same: reference only. The arithmetic is small
  and is already reimplemented bit exact.

## What we would have to write ourselves

- Loader, consumer and storer bodies for every non-GEMM operation.
- The page plan for each operation: which pages it needs, in what order it
  releases them. Only five pages (80 KiB) exist, so this is the tightest
  design constraint.
- The semaphore protocol inside each operation, and a progress argument that
  it cannot deadlock.
- Fine-grained cross-SM dependency counters, and the compiler logic that
  emits them, replacing the per-stage wave barrier.
- A compiler that assigns tiles to SMs per instruction, since SMs would no
  longer all run the identical list.

## Potential benefits

- **Overlap.** Loading, computing and storing proceed together, and the next
  instruction's inputs arrive while the current one computes.
- **Cleaner compute warps.** Memory operations were the measured cost in the
  attention; moving them to a loader leaves consumers doing arithmetic only.
- **Fewer global stalls.** Per-tile dependencies instead of 8,000 waits for
  every SM.
- **Per-role timing.** The runtime can record when each role starts and
  finishes each instruction, which the all-warp wrapper discards.

## What it does not fix, and the risks

- **The register limit stays.** On SM120 the assembler rejects the
  instruction that gives roles different register budgets, so all 384
  threads share the 168-register cap whatever their role. Splitting roles
  helps by putting less code in each warp, not by raising the cap.
- **Fewer compute warps.** Eight consumers instead of twelve. This only pays
  off if the tensor cores, not warp count, are the limit.
- **Much harder to get right.** Semaphore and page bugs show up as hangs,
  and every test is a remote run.
- **Unproven here.** None of this has been built or measured in this
  project. The benefits above are the design's intent, not results.

## Suggested path

1. Keep the all-warp GEMM and the all-warp row-wise operations.
2. Convert the attention alone to loader / consumer / storer with a
   double-buffered tile page. It is the bottleneck and the best fit.
3. Measure with `megakernel/tests/dit_attention_bench.py` against the
   current 459 ms and against the parity test.
4. Convert further operations only if that measurement justifies it.
