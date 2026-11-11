### Profiling on Modal

This guide is for running Nsight Systems and Nsight Compute on a Modal GPU and reading the reports on your own laptop. If you have used them on a laptop with a GPU, the tools are the same. The difference is where they run: the GPU sits in a Modal container, so we run the command-line tools there, save the report to a Volume, copy it down and open it in the viewer.

We profile our baseline: [FastH3 V2 NVFP4 Consumer](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer) at `9d0d0d3` on FastVideo at `ce970eb`, one 1344x768, 5 s request, every model on one RTX PRO 6000.

Modal runs containers under gVisor, so two defaults have to change. ncu cannot lock the GPU clocks, so it needs `--clock-control none`. nsys cannot sample the CPU, so it needs `--sample=none --cpuctxsw=none`. `baseline/fastvideo_modal.py` already passes both.

### Step 1: Set up once

Each of us runs in our own Modal workspace, so GPU time and storage bill to your account. Install the Modal CLI, log in and check which workspace you are in.

```bash
pip install modal
modal setup
modal profile current
```

Clone our repo and a FastVideo checkout with its two kernel submodules. The image is built from that checkout, so every `modal run` below needs `H3MK_FV_SRC` set to it.

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
git checkout ce970eb
git submodule update --init fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk
export H3MK_FV_SRC=$PWD
```

On Windows PowerShell, use `$env:H3MK_FV_SRC = "C:\path\to\FastVideo"`. The first run builds the image, which compiles FastVideo's CUDA kernels. Later runs reuse it.

From our repo, download the checkpoint into a Volume named `h3-megakernel-cache`, which the script creates. The model is public, so no Hugging Face token is needed. The repo is 63 GB, because it also ships the BF16 DiT shards and the dense VAE. At run time FastVideo takes the 350 big DiT linears from `transformer/nvfp4_weights.safetensors` (11.9 GB) and 156 video decoder linears from `vae/minimax_h3_video_vae_int8_convrot.safetensors` (2.1 GB).

```bash
modal run --detach baseline/fastvideo_modal.py::download
modal volume ls h3-megakernel-cache hf/hub
```

When the download finishes, the second command lists `models--FastVideo--FastVideo-FastH3-8-Step-V2-NVFP4-Consumer`.

Install the viewers on your laptop: [Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started) 2025.3.2 or newer and [Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started) 2025.3.1 or newer, the versions in our image. Viewing needs no GPU. On a Mac, take the arm64 build; its Nsight Systems also ships the `nsys` command line at `/Applications/NVIDIA Nsight Systems.app/Contents/MacOS/nsys`.

### Step 2: Record one request with nsys

Run this from our repo. Each run needs a new `--run-id`, which becomes its folder `runs/<run-id>/` on the Volume.

```bash
modal run --detach baseline/fastvideo_modal.py::go --run-id nsys-1344x768-<you> --args "--shapes 1344x768 --reps 2 --profile-rep 1" --nsys
```

The script runs two requests. The first one compiles the VAE for about 4 minutes, so `--profile-rep 1` wraps only the second one in `cudaProfilerStart/Stop`. It also sets `FASTVIDEO_NVTX_PROFILE=1`, which labels every stage, step and block. A run takes about 8 minutes, at about $3 an hour for the GPU. `--detach` keeps it going if you close the terminal; the printed link shows the logs.

This table shows the nsys options the script adds.

| Option | Why |
|---|---|
| `--capture-range=cudaProfilerApi` | record only the second request |
| `--trace=cuda,nvtx` | kernels, CUDA calls and FastVideo's labels |
| `--cuda-graph-trace=node` | the VAE replays CUDA graphs; this shows the kernels inside them |
| `--sample=none --cpuctxsw=none` | gVisor has no CPU sampling |

When it finishes, the terminal prints `returncode 0`, and the run folder holds `request.nsys-rep`, `request.sqlite` and two CSV summaries.

### Step 3: Copy the nsys report to your laptop

Copy the report into `build/profiles/` in our repo, which git ignores. Make the folder first: if it does not exist, `modal volume get` saves the report as a file with the folder's name.

```bash
modal volume ls h3-megakernel-cache runs/nsys-1344x768-<you>
mkdir -p build/profiles/nsys-1344x768-<you>
modal volume get h3-megakernel-cache runs/nsys-1344x768-<you>/request.nsys-rep build/profiles/nsys-1344x768-<you>/
```

### Step 4: Open it in Nsight Systems

1. Start Nsight Systems: from the Start menu on Windows, from Applications on a Mac (`NVIDIA Nsight Systems`), or with `nsys-ui` on Linux.
2. Choose File > Open and pick `build/profiles/nsys-1344x768-<you>/request.nsys-rep`. On a Mac you can also run `open -a "NVIDIA Nsight Systems" build/profiles/nsys-1344x768-<you>/request.nsys-rep`.
3. Wait a few seconds for the timeline to load.
4. In the tree on the left, expand `CUDA HW` to see the kernels on the GPU.
5. Expand `Threads` and find the thread with an `NVTX` row under it. That is FastVideo's forward thread. Its top ranges are `minimax_h3.text_encoding`, `minimax_h3.dit`, `minimax_h3.vae` and `minimax_h3.audio_vae`. Hover over a range to see its length.
6. Inside the DiT are `minimax_h3.step0` to `step7`, then `minimax_h3.transformer_block.0` to `49`, then `h3.*` ranges for each part of a block. To look at one block, drag across it on the timeline and zoom to the selection from the right-click menu.

The same numbers come from `nsys stats`, run in the report's folder on your laptop or in the shell from step 7. `Total Proj Time` is the GPU time of a range and `Total Range Time` is its CPU time. `/4` picks the fifth instance of a range, here block 25 in `minimax_h3.step4`. If nsys says an existing SQLite export is older than the report, add `--force-export=true` once.

```bash
nsys stats -r nvtx_gpu_proj_sum request.nsys-rep
nsys stats -r cuda_gpu_kern_sum --filter-nvtx minimax_h3.dit request.nsys-rep
nsys stats -r cuda_gpu_kern_sum --filter-nvtx "minimax_h3.transformer_block.25/4" request.nsys-rep
```

### Step 5: Profile kernels with ncu

ncu replays each kernel it profiles about 40 times, so give it a filter, or it profiles every one of the request's 54,000 or so kernels. `--profile-from-start off` is already set, so only the second request counts. Pick launches with `-k` (kernel name), `-s` (skip) and `-c` (count), or with an NVTX range. Count them from the nsys report: each step runs 50 attention kernels, and each block's MLP runs 7 kernels, so `-s 225` and `-s 1575` both land on block 25 in `minimax_h3.step4`.

```bash
modal run --detach baseline/fastvideo_modal.py::go --run-id ncu-attn-<you> --args "--shapes 1344x768 --reps 2 --profile-rep 1" --ncu "-k regex:compute_attn_ws -s 225 -c 1 --set full --import-source yes"
modal run --detach baseline/fastvideo_modal.py::go --run-id ncu-ffn-<you> --args "--shapes 1344x768 --reps 2 --profile-rep 1" --ncu "--nvtx --nvtx-include minimax_h3.transformer_block.feed_forward/ -s 1575 -c 7 --set full --import-source yes"
```

Each run takes about 8 minutes and writes `runs/<run-id>/kernels.ncu-rep`. Copy it down the same way as in step 3.

```bash
mkdir -p build/profiles/ncu-attn-<you>
modal volume get h3-megakernel-cache runs/ncu-attn-<you>/kernels.ncu-rep build/profiles/ncu-attn-<you>/
```

### Step 6: Open it in Nsight Compute

1. Start Nsight Compute: from the Start menu on Windows, from Applications on a Mac (`NVIDIA Nsight Compute`), or with `ncu-ui` on Linux.
2. Choose File > Open File and pick `build/profiles/ncu-attn-<you>/kernels.ncu-rep`. On a Mac you can also run `open -a "NVIDIA Nsight Compute" build/profiles/ncu-attn-<you>/kernels.ncu-rep`.
3. The report opens on the Summary page, one row per profiled kernel. The attention report has one kernel and the MLP report has seven.
4. Pick a kernel from the Result list at the top, then switch the Page list to Details.
5. Read the sections from the top: `GPU Speed Of Light Throughput`, `Memory Workload Analysis`, `Occupancy` and `Warp State Statistics`. The Source page shows the SASS with stalls per instruction.

Since the clocks are not locked and ncu flushes the caches before each kernel, its times run a little high. Use ncu to find the limiter and nsys for time.

### Step 7: Work in a shell on the GPU

For anything the script does not cover, open a shell in the same container: same image, the RTX PRO 6000, `nsys` and `ncu` on the path, the Volume at `/cache` and our scripts in `/root/baseline`.

```bash
modal shell baseline/fastvideo_modal.py::run
```

Then use the tools as on your laptop. This command profiles the same attention kernel as step 5:

```bash
mkdir -p /cache/runs/shell-<you> && cd /root/baseline
OUT_DIR=/cache/runs/shell-<you> ncu --clock-control none --profile-from-start off -k regex:compute_attn_ws -s 225 -c 1 --set full -o /cache/runs/shell-<you>/attn python fastvideo_capacity.py --shapes 1344x768 --reps 2 --profile-rep 1
```

Files under `/cache` are saved when you leave the shell and show up in `modal volume ls` a few seconds later. The shell ends as soon as your terminal disconnects, even in the middle of a command, so use `modal run --detach` from steps 2 and 5 for long runs. The GPU is billed while the shell is open, so exit when you are done.

To profile your own script, copy three things from `baseline/fastvideo_modal.py`: install `cuda-nsight-systems-13-0` and `cuda-nsight-compute-13-0` in a CUDA 13.0 image, add the two gVisor options, and write the report to the Volume. A script in `baseline/` needs none of that: pass it to `go` with `--script`, and `--nsys` or `--ncu` work as above. For example, `--script fi_mlp_bench.py --args "--profile 4096" --nsys` records the MLP block from [Milestone 1](milestones/m1.md).

### Problems we hit

This table shows the errors we saw and their fixes.

| Symptom | Fix |
|---|---|
| ncu stops with `Failed to lock GPU clock frequencies` | add `--clock-control none` |
| nsys fails on CPU sampling | add `--sample=none --cpuctxsw=none` |
| ncu runs for hours | filter with `-k`, `-s`, `-c` or `--nvtx-include` |
| ncu prints `No kernels were profiled` | the regex or skip count is off; check the counts in nsys |
| the VAE shows up as one block on the timeline | add `--cuda-graph-trace=node` |
| the run stops at once with `FileExistsError` | use a new `--run-id` |
| the downloaded report is a file without an extension | make the local folder before `modal volume get` |
| a `modal shell` run stops halfway | the terminal disconnected; keep it open or use `modal run --detach` |
| `nsys stats` says the SQLite export is older | add `--force-export=true` |
