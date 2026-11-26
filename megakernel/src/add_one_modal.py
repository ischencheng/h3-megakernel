import sys
import modal
from pathlib import Path


# Modal location from .add_local_dir
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root/h3mk")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("ninja")  # torch.utils.cpp_extension.load needs it
    .env({"CUDA_HOME": "/usr/local/cuda"})
    # same layout as the repo, so add_one.py's include paths still work
    .add_local_dir(ROOT / "megakernel", "/root/h3mk/megakernel", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "ThunderKittens" / "include", "/root/h3mk/ThunderKittens/include")
)
app = modal.App("add_one_test")

@app.function(image= image, gpu="RTX-PRO-6000", timeout= 1800)
def test():
    import subprocess, sys
    subprocess.run([sys.executable, "/root/h3mk/megakernel/src/add_one.py"], check=True)

@app.local_entrypoint()
def main():
    test.remote()
    print("test complete")
