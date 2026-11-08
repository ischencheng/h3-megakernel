# H3 Megakernel

CIS 5650 final project: run the MiniMax-H3 text-to-video-and-audio pipeline
(Qwen3-VL text encoder, DiT, video and audio decoders) in one CUDA launch on
one RTX PRO 6000, and measure what fusion buys at each stage. Our hypothesis is
that fusion helps less as kernels get bigger.

## Milestone 1: validate one component

The component is the MLP of one DiT block: `y = (silu(x W_gate^T) * (x W_up^T)) W2^T`.

| Tensor | Shape | Notes |
|---|---|---|
| `x` | `[M, 5376]` BF16 | norm2 + AdaLN output |
| `w1` | `[28672, 5376]` BF16 | rows `[0, 14336)` gate, `[14336, 28672)` up |
| `w2` | `[5376, 14336]` BF16 | |
| `y` | `[M, 5376]` BF16 | |

We compare a split build (two kernels) and a fused build (one persistent
kernel) that share the same GEMM tile, so any difference comes from fusion.
The token counts run from 256 to 37,696; 14,830 and 37,696 are the DiT
lengths of a 5 s video at 832x480 and 1344x768 (derivation in `m1/config.py`).

What is in this repo:

| Path | What it does |
|---|---|
| `m1/config.py` | Shapes, token counts, timing and pass rules, pinned checkpoint |
| `m1/weights.py` | Reads only the block-0 MLP weights (about 0.46 GB) from the HF shard by byte range |
| `m1/reference.py` | FP32 reference (TF32 off) and the cuBLAS BF16 path |
| `m1/check.py` | Error metrics and the pass rule |
| `m1/bench.py` | CUDA-event timing with L2 flushed before each run, CUDA graph capture |
| `m1/kernels.py` | Builds `csrc/` as a PyTorch extension and lists the variants |
| `m1/run.py` | Checks every variant, times the ones that pass, writes `results/*.json` |
| `csrc/mlp_split.cu` | Split build (encoder owner), not written yet |
| `csrc/mlp_fused.cu` | Fused build (DiT owner), not written yet |
| `modal_run.py` | Runs `m1.run` on a Modal GPU |

### Weight layout

The Diffusers export stores `ff.net.0.proj` value-first (up, then gate);
H3's native `fc1` is gate-first, and vLLM-Omni swaps the halves when loading.
`m1/weights.py` saves `w1` already gate-first, so kernels and the reference
agree.

### Rules

- **Correctness:** a variant passes if its `rel_l2` and `max_over_rms` against the
  FP32 reference are at most 2x those of the cuBLAS BF16 path on the same input.
  Variants that fail are not timed.
- **Timing:** median of 20 runs after 3 warmups, with a 256 MiB buffer written
  before each run to flush the 128 MiB L2. Each result also reports p10, p90 and spread.
- **Inputs:** random BF16 at unit scale for now. The MLP works token by token, so
  real activations, which come with the vLLM-Omni baseline in Milestone 2, should
  not change the comparison.

### Run

```bash
modal run modal_run.py                                   # RTX PRO 6000, full sweep
modal run modal_run.py --gpu A10 --args "--tokens 256 1024"
modal run modal_run.py --args "--variants cublas cublas_graph split fused_g2048"
modal run modal_run.py --ncu "--tokens 4096 --variants cublas split"
```

Use A10 while writing kernels. It has the same per-SM limits as the RTX PRO
6000 (100 KiB shared memory, 99 KiB per block, 64K registers, 1,536 threads)
at about a third of the price, but only a 6 MiB L2, so record timings on the
RTX PRO 6000 only.

`--ncu` runs each variant once inside an NVTX range and collects kernel time,
DRAM bytes and L2 bytes. On sm_120 the DRAM counters are
`dram__bytes_op_read` and `dram__bytes_op_write` (Hopper calls them
`dram__bytes_read` and `dram__bytes_write`). Modal containers need
`--clock-control none`, which the script already passes.

The `cublas` variants run cuBLAS for both GEMMs and one fused SwiGLU kernel
(via `torch.compile`), matching the structure of vLLM-Omni's eager MLP.

### Variants

| Name | What |
|---|---|
| `cublas`, `cublas_graph` | PyTorch BF16 path, plain and as a CUDA graph |
| `split`, `split_graph` | Our two kernels, plain and as a CUDA graph |
| `fused_g1024`, `fused_g2048`, `fused_g4096` | Our persistent kernel with token groups of 1024, 2048 and 4096 |

A group's intermediate is 29, 59 and 117 MB, against a 128 MiB L2. Smaller
groups keep `h` in L2 but re-read the 0.46 GB of weights more often.

Variants not written yet show `not_implemented`.

### First numbers: cuBLAS on one RTX PRO 6000 (2026-11-08)

From `results/m1-RTX-PRO-6000-Blackwell-Server-Edition-20261108-060852.json`
and one `--ncu` run. This is the bar our kernels are compared against.

| Tokens | Median (ms) | TFLOP/s | FC1 DRAM read | FC2 DRAM read |
|---:|---:|---:|---:|---:|
| 256 | 0.472 | 251 | 0.31 GB | 0.16 GB |
| 1,024 | 1.329 | 356 | | |
| 4,096 | 4.820 | 393 | 0.39 GB | 0.36 GB |
| 14,830 | 18.418 | 372 | | |
| 37,696 | 46.316 | 376 | 5.99 GB | 4.98 GB |

Times are the CUDA-graph variant with L2 flushed. The cuBLAS kernels here are
`cutlass_80_tensorop_bf16_s16816gemm` (`mma.sync` with 256x128 tiles), since
sm_120 has no `wgmma` or `tcgen05`. At 37,696 tokens FC1 reads 5.99 GB from
DRAM although its weights are only 0.31 GB, so cuBLAS already re-reads the
weights many times; the fused build's group size has to be judged against that.
