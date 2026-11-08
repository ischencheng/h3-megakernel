University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Final Project - H3 Megakernel

* Chen Cheng, Ryan Morris, Neel Shejwalkar
* Tested on: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB

### Screenshots

![fasth3_v2_1344x768](images/fasth3_v2_1344x768.png)

These are four frames of a 5 s clip with audio, generated on one RTX PRO 6000.

### H3 Megakernel

MiniMax-H3 turns a text prompt into video with sound. It has a text encoder, a DiT, a video decoder and an audio decoder. Our goal is to run all four in one CUDA launch and measure what fusion buys at each stage. We expect fusion to help less as kernels get bigger.

### Checkpoint

We use [FastH3 V2 NVFP4 Consumer](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer) at `9d0d0d3` and run it with [FastVideo](https://github.com/hao-ai-lab/FastVideo) at `ce970eb`. It is H3 distilled to 8 steps, with most large linears stored in NVFP4, a 4-bit format with one scale per 16 values.

The text encoder is Qwen3-VL cut to 50 layers. The DiT has 50 blocks with sparse attention on 64-token tiles. The video decoder is a 26-block lightweight VAE with INT8 weights. We generate 1344x768, 5 s clips (124 frames) with 8 forwards and no CFG.

FastVideo offloads models to the host by default. We turn that off, because one launch needs every model on the GPU.

### Running

The whole pipeline needs a FastVideo checkout at `ce970eb` with the `cutlass` and `tk` submodules under `fastvideo-kernel/include/`. Use one shape per process, because a second shape fails while compiling its VAE graph on a new thread.

```bash
H3MK_FV_SRC=<checkout> modal run --detach baseline/fastvideo_modal.py::download
H3MK_FV_SRC=<checkout> modal run --detach baseline/fastvideo_modal.py::go --run-id <id> --args "--shapes 1344x768"
```

### Performance Analysis

We measured everything on one RTX PRO 6000 with 97,887 MiB of memory. Memory comes from `nvidia-smi` every 0.5 s. The first request compiles the VAE and takes about 4 minutes, so the times below are later requests, and those stayed within 1% of each other.

This table shows the whole pipeline with every model on the GPU.

| | 1344x768 | 832x480 |
|---:|---:|---:|
| Memory after loading | 58,275 MiB | 58,273 MiB |
| Peak memory | 67,851 MiB | 62,797 MiB |
| Text encoder | 0.09 s | 0.09 s |
| DiT, 8 forwards | 25.9 s | 8.7 s |
| Video decoder | 4.5 s | 1.9 s |
| Audio decoder | 0.05 s | 0.05 s |
| Whole request | 31.0 s | 11.0 s |

From the table, the DiT is 84% of a 1344x768 request and the video decoder is 14%. The first request peaks at 68,385 MiB, which leaves 29,502 MiB free. Our kernels' buffers have to fit in that, and we have not measured them yet.

### Credits

* [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and [FastH3](https://huggingface.co/collections/FastVideo/fastvideo-fasth3) for the model weights
* [FastVideo](https://github.com/hao-ai-lab/FastVideo) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for the baseline
* [LynnReal lightweight VAE](https://github.com/LynnReal-AI/LynnReal-Omni) and [Kijai's INT8 weights](https://huggingface.co/Kijai/MiniMax-H3-experimental) for the video decoder
