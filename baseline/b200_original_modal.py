"""Download original MiniMax H3 on CPU, then generate one clip on a B200."""

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, "/root/baseline")
from fastvideo_modal import HERE, image, run_script, volume
# Root T2VA checkpoint. FL2VA/ and Ref2VA/ are task partitions and lack the schedulers.
T2VA_PATTERNS = ["model_index.json", "modular_model_index.json", "scheduler/**", "audio_scheduler/**", "vae/**", "audio_vae/**", "text_encoder/**", "tokenizer/**", "processor/**", "transformer/**"]

app = modal.App("h3mk-b200-original-h3")
MODEL = "MiniMaxAI/MiniMax-H3"
REVISION = "0919fb684102ef115126612f3fbb84d1027e7317"


@app.function(image=image, cpu=16, memory=32768, volumes={"/cache": volume}, timeout=7200, retries=0)
def prefetch():
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=T2VA_PATTERNS))
    missing = [d for d in ("scheduler", "audio_scheduler", "vae", "audio_vae", "transformer") if not (root / d).is_dir()]
    if missing or not (root / "model_index.json").exists():
        raise FileNotFoundError(f"Original H3 T2VA download incomplete: {missing}")
    volume.commit()
    print(f"Original BF16 H3 T2VA cached at {root}", flush=True)


@app.function(image=image, gpu="B200", cpu=8, memory=65536,
              volumes={"/cache": volume}, timeout=7200, retries=0)
def benchmark(run_id: str, steps: int):
    if steps not in (20, 50):
        raise ValueError("Expected 20 or 50 original-H3 sampling steps")
    return run_script(run_id, "b200_original_h3.py", ["--steps", str(steps)])


@app.local_entrypoint()
def cache():
    prefetch.remote()


@app.local_entrypoint()
def go(run_id: str, steps: int):
    prompts = json.loads((HERE / "prompts.json").read_text())
    if len(prompts) != 1 or "basketball" not in prompts[0].lower():
        raise ValueError("Expected exactly one basketball prompt")
    rc = benchmark.remote(run_id, steps)
    print(f"original H3 returncode: {rc}")
    if rc:
        raise SystemExit(rc)
