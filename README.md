University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Final Project - H3 Megakernel

* Chen Cheng, Ryan Morris, Neel Shejwalkar
* Tested on: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB

### Demo

https://github.com/user-attachments/assets/5839d2a3-41d2-468b-96dc-6bdff2e7ea01

This is a 5 s clip at 1344x768 with audio, generated on one RTX PRO 6000.

### H3 Megakernel

MiniMax-H3 turns a text prompt into video with sound. It has a text encoder, a DiT, a video decoder and an audio decoder. Our goal is to run all four in one CUDA launch and measure what fusion buys at each stage. We expect fusion to help less as kernels get bigger.

### Milestones

| Milestone | Date | Goal |
|---|---|---|
| [Milestone 1](docs/milestones/m1.md) | 11/11 | Profile the baseline and run the DiT MLP block in one launch inside FastVideo |
| [Milestone 2](docs/milestones/m2.md) | 11/18 | Do the same for the attention half of the DiT, the text encoder and the decoders, and compare each with FastVideo |
| [Milestone 3](docs/milestones/m3.md) | 11/30 | Integrate all stages into one megakernel |
| [Final](docs/milestones/final.md) | 12/6 | Test different prompts and report the results |

The hand-offs between stages are in [docs/contract.md](docs/contract.md).

### Layout

```
docs/       milestones and the stage contract
baseline/   runs FastVideo, swaps our stages into it and benchmarks them
harness/    checkpoint weights, the FP32 reference, error checks and the L2 flush
stages/     text_encoder, dit, video_decoder, audio_decoder
skeleton/   the final single launch
results/    measured numbers
FastVideo/  FastVideo at ce970eb, as a submodule
third_party/  HuggingArch with an H3 overlay, as a submodule
```

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

In Milestone 1 we replaced the MLP half of every DiT block. The request went from 30.84 s to 29.10 s. FlashInfer's fused kernels account for 1.71 s of that, and running them as one launch adds 0.03 s. Details are in [Milestone 1](docs/milestones/m1.md).

In Milestone 2 we ran the video decoder's blocks as eight fused kernels each. The decoder went from 4.45 s to 2.61 s at 1344x768 and from 1.89 s to 1.12 s at 832x480, and the 1344x768 request from 30.9 s to 29.0 s. Details are in [Milestone 2](docs/milestones/m2.md#video-decoder).

### Credits

* [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and [FastH3](https://huggingface.co/collections/FastVideo/fastvideo-fasth3) for the model weights
* [FastVideo](https://github.com/hao-ai-lab/FastVideo) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for the baseline
* [LynnReal lightweight VAE](https://github.com/LynnReal-AI/LynnReal-Omni) and [Kijai's INT8 weights](https://huggingface.co/Kijai/MiniMax-H3-experimental) for the video decoder
