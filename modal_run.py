"""Run Milestone 1 on a Modal GPU and copy the results JSON back to results/.

  modal run modal_run.py                                   # RTX PRO 6000, full sweep
  modal run modal_run.py --gpu A10 --args "--tokens 256 1024"
  modal run modal_run.py --ncu "--tokens 4096 --variants cublas split"

Use A10 while writing kernels: it has the same per-SM limits as the RTX PRO
6000 (100 KiB shared memory, 64K registers, 1,536 threads) at a third of the
price, but only a 6 MiB L2, so record timings on the RTX PRO 6000 only.
"""

import glob
import shlex
import shutil
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("cuda-nsight-compute-13-0")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --index-url https://download.pytorch.org/whl/cu130 'torch==2.13.0'",
        "uv pip install --system 'safetensors==0.8.0' 'numpy==2.5.2' 'ninja==1.11.1.4'",
    )
    .env({"H3MK_DATA": "/tmp/h3mk-data", "H3MK_RESULTS": "/tmp/h3mk-results",
          "TORCH_EXTENSIONS_DIR": "/tmp/torch-extensions"})
    .add_local_dir(REPO, "/root/h3mk",
                   ignore=["data", "results", ".git", "**/__pycache__", ".agents", ".claude"])
)

app = modal.App("h3mk-m1")


@app.function(image=image, gpu="RTX-PRO-6000", timeout=3600)
def run(argv: list[str], commit: str, ncu: bool):
    import os

    env = dict(os.environ, H3MK_COMMIT=commit)
    cmd = ["python", "-m", "m1.run", *argv]
    if ncu:
        exe = shutil.which("ncu") or sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
        cmd = [exe, "--clock-control", "none", "--nvtx", "--nvtx-include", "mlp/", "--csv",
               # sm_120 names; Hopper calls these dram__bytes_read/write
               "--metrics", "gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum",
               *cmd, "--profile-once"]
    proc = subprocess.run(cmd, cwd="/root/h3mk", env=env, capture_output=True, text=True)
    results = {p.name: p.read_text() for p in Path("/tmp/h3mk-results").glob("*.json")}
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr[-20000:],
            "results": results}


def _commit():
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return "uncommitted"
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True,
                                    stderr=subprocess.DEVNULL).strip()
    return head + ("-dirty" if dirty else "")


@app.local_entrypoint()
def main(gpu: str = "RTX-PRO-6000", args: str = "", ncu: str = ""):
    argv = shlex.split(ncu or args)
    out = run.with_options(gpu=gpu).remote(argv, _commit(), bool(ncu))
    print(out["stdout"])
    if out["returncode"] != 0:
        print(out["stderr"])
        raise SystemExit(f"m1.run exited with {out['returncode']}")
    results = REPO / "results"
    results.mkdir(exist_ok=True)
    for name, text in out["results"].items():
        (results / name).write_text(text)
        print(f"saved results/{name}")
