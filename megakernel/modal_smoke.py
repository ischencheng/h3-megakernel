"""Build and run the Hazy smoke test on SM100 or SM120."""
from pathlib import Path
import subprocess

import modal


ROOT = Path(__file__).resolve().parent
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential")
    .pip_install("torch==2.12.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("pybind11")
    .add_local_dir(str(ROOT), "/root/megakernel", ignore=[".git/**", "**/__pycache__/**"])
)
app = modal.App("h3mk-architecture-smoke")


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=16384, timeout=1800)
def verify_sm120():
    subprocess.run(["make", "GPU=RTXPRO6000", "smoke"], cwd="/root/megakernel", check=True)
    subprocess.run(["make", "GPU=RTXPRO6000", "graph"], cwd="/root/megakernel", check=True)
    return "SM120 smoke and graph passed"


@app.function(image=image, gpu="B200", cpu=8, memory=16384, timeout=1800)
def verify_sm100():
    subprocess.run(["make", "GPU=B200", "smoke"], cwd="/root/megakernel", check=True)
    return "SM100 smoke passed"


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=16384, timeout=1800)
def verify_dual_build():
    root = Path("/root/megakernel")
    subprocess.run(["make", "GPU=B200", "all"], cwd=root, check=True)
    subprocess.run(["make", "GPU=RTXPRO6000", "smoke"], cwd=root, check=True)
    subprocess.run(["make", "GPU=B200", "all"], cwd=root, check=True)
    assert len(list((root / "build/B200").glob("mk_h3*.so"))) == 1
    assert len(list((root / "build/RTXPRO6000").glob("mk_h3*.so"))) == 1
    return "SM100 and SM120 build outputs coexist; SM120 smoke passed"


@app.local_entrypoint()
def main(gpu: str = "RTXPRO6000"):
    if gpu == "RTXPRO6000":
        print(verify_sm120.remote())
    elif gpu == "B200":
        print(verify_sm100.remote())
    elif gpu == "BOTH":
        print(verify_dual_build.remote())
    else:
        raise ValueError("gpu must be RTXPRO6000, B200, or BOTH")
