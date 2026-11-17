import glob
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SRC = os.environ.get("H3MK_FV_SRC", "")
MODEL = "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer"
REVISION = "9d0d0d3ae69365aded83fab5562f00f9f06c12aa"

volume = modal.Volume.from_name("h3-megakernel-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "ffmpeg", "curl", "build-essential")
    .pip_install("uv")
    .env({"UV_SYSTEM_PYTHON": "1", "UV_TORCH_BACKEND": "cu130", "CUDA_HOME": "/usr/local/cuda",
          "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0"})
    .add_local_dir(SRC or str(HERE), "/FastVideo", copy=True,
                   ignore=[".git/**", "docs-site/**", "assets/**", "comfyui/**", "apps/**", "**/__pycache__/**"])
    .run_commands(
        "cd /FastVideo && uv pip install -e .",
        "cd /FastVideo/fastvideo-kernel && CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST=12.0a CMAKE_BUILD_PARALLEL_LEVEL=4 CMAKE_ARGS= "
        "./build.sh --wheel-dir /opt/fastvideo-kernel-wheel",
        "ls -la /opt/fastvideo-kernel-wheel",  # fastvideo can only be imported on a gpu
    )
    # flashinfer needs nvidia-ml-py, and pynvml shadows it
    .run_commands("uv pip uninstall pynvml || true", "uv pip install --reinstall nvidia-ml-py")
    .apt_install("cuda-nsight-systems-13-0", "cuda-nsight-compute-13-0")
    .run_commands("uv pip install ninja")  # flashinfer's jit builds the one-launch mlp (stages/dit/fi_mlp.py)
    # H3MK_EXT_DIR keeps the decoder kernels' build (stages/video_decoder) across runs
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1", "H3MK_DATA": "/tmp/h3mk-data",
          "H3MK_EXT_DIR": "/cache/h3mk-ext"})
    .add_local_dir(HERE, "/root/baseline", ignore=["**/__pycache__/**"])
    .add_local_dir(HERE.parent / "harness", "/root/h3mk/harness", ignore=["**/__pycache__/**"])
    .add_local_dir(HERE.parent / "stages", "/root/h3mk/stages", ignore=["**/__pycache__/**"])
)
app = modal.App("h3mk-fastvideo")


def _tool(name):
    return shutil.which(name) or sorted(glob.glob(f"/opt/nvidia/nsight-*/*/{name}") +
                                        glob.glob(f"/opt/nvidia/nsight-*/*/bin/{name}"))[-1]


@app.function(image=image, cpu=16, memory=32768, volumes={"/cache": volume}, timeout=7200, retries=1)
def download():
    from huggingface_hub import snapshot_download

    print(snapshot_download(MODEL, revision=REVISION), flush=True)
    volume.commit()


@app.function(image=image, gpu="RTX-PRO-6000", cpu=(8, 32), memory=(131072, 393216),
              volumes={"/cache": volume}, timeout=7200, retries=0)
def run(run_id: str, script: str, argv: list[str], nsys: bool = False, ncu: str = ""):
    out = Path("/cache/runs") / run_id
    out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()

    def sync():
        while not stop.wait(120):
            volume.commit()

    threading.Thread(target=sync, daemon=True).start()
    env = {**os.environ, "OUT_DIR": str(out), "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root/baseline:/root/h3mk"}
    cmd = [sys.executable, f"/root/baseline/{script}", *argv]
    # both record only what runs between cudaProfilerStart and Stop (--profile-rep)
    if nsys:
        # gvisor has no perf_event_open, so no cpu sampling; node trace opens the vae's cuda graphs
        cmd = [_tool("nsys"), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
               "--output", str(out / "request"), *cmd]
    elif ncu:
        # modal cannot lock the gpu clocks
        cmd = [_tool("ncu"), "--clock-control", "none", "--profile-from-start", "off",
               "--export", str(out / "kernels"), *shlex.split(ncu), *cmd]
    with open(out / "driver.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        proc.wait()
    if nsys and (out / "request.nsys-rep").exists():
        subprocess.run([_tool("nsys"), "stats", "--report", "nvtx_gpu_proj_sum,cuda_gpu_kern_sum", "--format", "csv",
                        "--output", str(out / "request"), str(out / "request.nsys-rep")])
    stop.set()
    (out / "returncode").write_text(str(proc.returncode))
    volume.commit()
    return proc.returncode


@app.local_entrypoint()
def go(run_id: str, script: str = "fastvideo_capacity.py", args: str = "", nsys: bool = False, ncu: str = ""):
    print("returncode", run.remote(run_id, script, shlex.split(args), nsys, ncu))
