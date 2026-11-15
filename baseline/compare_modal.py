"""Run three quality recipes sequentially; download on CPU before allocating a GPU."""
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

# Modal uploads this entrypoint at /root, while its companion scripts live here.
if Path("/root/baseline").is_dir():
    sys.path.insert(0, "/root/baseline")

from convrot import COMFY_COMMIT, REPO, REVISION, model_files
from fastvideo_modal import HERE, MODEL, REVISION as FAST_REVISION, SRC, image, run_script, volume

app = modal.App("h3mk-compare")
VARIANTS = ("fast-fp4", "fast-no-fp4", "convrot")
comfy_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "ffmpeg", "build-essential")
    .pip_install("uv")
    .env({"UV_SYSTEM_PYTHON": "1", "HF_HOME": "/cache/hf"})
    .run_commands(
        "git clone https://github.com/Comfy-Org/ComfyUI.git /ComfyUI",
        f"git -C /ComfyUI checkout {COMFY_COMMIT}",
        "uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.12.0 torchvision torchaudio",
        "uv pip install -r /ComfyUI/requirements.txt huggingface_hub",
    )
    .add_local_dir(HERE, "/root/baseline", ignore=["**/__pycache__/**"])
)
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir(HERE, "/root/baseline", ignore=["**/__pycache__/**"])
)


@app.function(image=download_image, volumes={"/cache": volume}, timeout=7200)
def download(variants: list[str]):
    from huggingface_hub import snapshot_download
    if any(v.startswith("fast-") for v in variants):
        snapshot_download(MODEL, revision=FAST_REVISION)
    if "convrot" in variants:
        snapshot_download(REPO, revision=REVISION,
                          allow_patterns=[f"{folder}/{name}" for folder, name in model_files()])
    volume.commit()


def execute(run_id, variant, shape, seeds, steps):
    from pathlib import Path
    argv = ["--prompts", "/root/baseline/prompts.json", "--shapes", shape,
            "--seeds", *map(str, seeds)]
    script = "convrot.py" if variant == "convrot" else "fastvideo_capacity.py"
    argv += ["--steps", str(steps)] if variant == "convrot" else ["--fp4-attention", str(int(variant == "fast-fp4"))]
    start = time.monotonic()
    try:
        code = run_script(f"{run_id}/{variant}", script, argv)
        if code:
            raise RuntimeError(f"{variant} failed (exit {code}); see runs/{run_id}/{variant}/driver.log")
    finally:
        out = Path("/cache/runs") / run_id / variant
        if out.exists():
            (out / "environment.txt").write_text(subprocess.check_output(["uv", "pip", "freeze"], text=True))
            (out / "wall.json").write_text(json.dumps({"process_wall_s": time.monotonic() - start}))
        volume.commit()


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=131072,
              volumes={"/cache": volume}, timeout=7200)
def fast(run_id: str, variant: str, shape: str, seeds: list[int], steps: int):
    execute(run_id, variant, shape, seeds, steps)


@app.function(image=comfy_image, gpu="RTX-PRO-6000", cpu=8, memory=131072,
              volumes={"/cache": volume}, timeout=7200)
def convrot(run_id: str, variant: str, shape: str, seeds: list[int], steps: int):
    execute(run_id, variant, shape, seeds, steps)


@app.function(image=download_image, timeout=43200)
def compare(run_id: str, selected: list[str], shape: str, seeds: list[int], steps: int):
    download.remote(selected)
    for variant in selected:
        worker = convrot if variant == "convrot" else fast
        worker.remote(run_id, variant, shape, seeds, steps)


@app.local_entrypoint()
def main(run_id: str = "", variants: str = ",".join(VARIANTS), shape: str = "1344x768",
         seeds: str = "1101,1102", steps: int = 20):
    selected = variants.split(",")
    run_id = run_id or datetime.now(timezone.utc).strftime("compare-%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or any(v not in VARIANTS for v in selected):
        raise ValueError(f"Use a simple run ID and variants from {VARIANTS}")
    width, height = map(int, shape.split("x"))
    seed_list = [int(seed) for seed in seeds.split(",")]
    prompts = json.loads((HERE / "prompts.json").read_text())
    if (not isinstance(prompts, list) or not prompts or
            any(not isinstance(p, str) or not p.strip() for p in prompts)):
        raise ValueError("prompts.json must contain a nonempty list of prompt strings")
    if len(set(seed_list)) != len(seed_list) or len(set(prompts)) != len(prompts):
        raise ValueError("Use distinct prompts and seeds so ComfyUI cannot reuse completed generations")
    if min(width, height, steps) <= 0 or width % 32 or height % 32 or min(seed_list) < 0:
        raise ValueError("Dimensions must be positive multiples of 32; steps positive; seeds nonnegative")
    if any(v.startswith("fast-") for v in selected):
        if not SRC:
            raise ValueError("Set H3MK_FV_SRC to the FastVideo checkout at ce970eb")
        commit = subprocess.check_output(["git", "-C", SRC, "rev-parse", "HEAD"], text=True).strip()
        if not commit.startswith("ce970eb"):
            raise ValueError(f"Expected FastVideo ce970eb; found {commit}")
    print(f"Outputs: h3-megakernel-cache/runs/{run_id}")
    compare.remote(run_id, selected, shape, seed_list, steps)
