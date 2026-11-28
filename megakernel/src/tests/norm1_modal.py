import sys
import modal
from datetime import datetime
from pathlib import Path


# Modal location from .add_local_dir
ROOT = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root/h3mk")
FV_COMMIT = "ce970eb71"  # FastVideo version the README pins

# FastVideo + FlashInfer install copied from baseline/fastvideo_modal.py
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .env({"UV_SYSTEM_PYTHON": "1", "UV_TORCH_BACKEND": "cu130", "CUDA_HOME": "/usr/local/cuda",
          "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0"})
    .run_commands(  # cloned in the image (cached by Modal), so nothing is uploaded per run
        f"git clone https://github.com/hao-ai-lab/FastVideo.git /FastVideo && cd /FastVideo && git checkout {FV_COMMIT}"
        " && git submodule update --init fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk",
        "cd /FastVideo && uv pip install -e .",
        "cd /FastVideo/fastvideo-kernel && CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST=12.0a CMAKE_BUILD_PARALLEL_LEVEL=4 CMAKE_ARGS= "
        "./build.sh --wheel-dir /opt/fastvideo-kernel-wheel",
    )
    # flashinfer needs nvidia-ml-py, and pynvml shadows it
    .run_commands("uv pip uninstall pynvml || true", "uv pip install --reinstall nvidia-ml-py")
    .run_commands("uv pip install ninja")  # torch.utils.cpp_extension.load needs it
    .add_local_dir(ROOT / "megakernel", "/root/h3mk/megakernel", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "ThunderKittens" / "include", "/root/h3mk/ThunderKittens/include")
)
app = modal.App("norm1_test")

@app.function(image= image, gpu="RTX-PRO-6000", timeout= 1800)
def test():
    import subprocess, sys
    r = subprocess.run([sys.executable, "/root/h3mk/megakernel/src/tests/norm1.py"], capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr, file=sys.stderr)
    return r.stdout + r.stderr, r.returncode  # container files vanish, so send the output back

@app.local_entrypoint()
def main():
    output, code = test.remote()
    results = ROOT / "megakernel" / "results"
    results.mkdir(exist_ok=True)
    out_file = results / f"norm1_{datetime.now():%Y%m%d_%H%M%S}.txt"  # one file per run
    out_file.write_text(output)
    print(f"opcode 3 NORMMODQUANT test {'complete' if code == 0 else 'FAILED'}, output in {out_file}")
