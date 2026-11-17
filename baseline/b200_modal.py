"""Two B200 H3 variants, exactly one basketball video from each."""

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, "/root/baseline")
from fastvideo_modal import HERE, image, run_script, volume

app = modal.App("h3mk-b200-one-prompt")
BF16_MODEL = "FastVideo/FastVideo-FastH3-8-Step-V2"


@app.function(image=image, cpu=16, memory=32768, volumes={"/cache": volume}, timeout=7200, retries=0)
def prefetch_bf16():
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(BF16_MODEL).sha
    print(f"BF16 checkpoint revision: {revision}", flush=True)
    path = snapshot_download(BF16_MODEL, revision=revision)
    revision_file = Path("/cache/models/bf16_v2_revision.txt")
    revision_file.parent.mkdir(parents=True, exist_ok=True)
    revision_file.write_text(revision)
    volume.commit()
    print(f"BF16 cached: {path}", flush=True)
    return revision


@app.function(image=image, gpu="B200", cpu=8, memory=(65536, 196608),
              volumes={"/cache": volume}, timeout=7200, retries=0)
def benchmark(variant: str, run_id: str):
    if variant not in ("consumer", "bf16"):
        raise ValueError(variant)
    return run_script(run_id, "b200_one_prompt.py", [variant])


@app.local_entrypoint()
def go(run_id: str, variant: str):
    prompts = json.loads((HERE / "prompts.json").read_text())
    if len(prompts) != 1 or not isinstance(prompts[0], str) or not prompts[0].strip():
        raise ValueError("Expected exactly one prompt in prompts.json")
    if variant == "bf16":
        prefetch_bf16.remote()
    rc = benchmark.remote(variant, run_id)
    print(f"{variant} returncode: {rc}")
    if rc:
        raise SystemExit(rc)


@app.local_entrypoint()
def cache_bf16():
    print(f"Pinned BF16 revision: {prefetch_bf16.remote()}")
