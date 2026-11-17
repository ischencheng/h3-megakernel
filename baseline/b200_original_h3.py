"""Exactly one original MiniMax H3 (root T2VA checkpoint) BF16 text-to-video generation."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from fastvideo_capacity import PeakMemory, describe

MODEL = "MiniMaxAI/MiniMax-H3"
REVISION = "0919fb684102ef115126612f3fbb84d1027e7317"
PROMPTS = Path("/root/baseline/prompts.json")
OUT = Path(os.environ["OUT_DIR"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=(20, 50), required=True)
    args = parser.parse_args()

    prompts = json.loads(PROMPTS.read_text())
    if len(prompts) != 1 or not isinstance(prompts[0], str) or "basketball" not in prompts[0].lower():
        raise ValueError("Expected exactly one basketball prompt in prompts.json")
    prompt = prompts[0]

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if "B200" not in gpu:
        raise RuntimeError(f"Expected B200, got {gpu}")

    from fastvideo import VideoGenerator
    from huggingface_hub import snapshot_download

    os.environ.update({
        "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
        "FASTVIDEO_MINIMAX_H3_DENSE_ATTN": "cudnn",
        "FASTVIDEO_H3_VAE_TILE_BATCH": "28",
        "FASTVIDEO_STAGE_LOGGING": "1",
    })
    print(f"GPU: {gpu}; original checkpoint: {MODEL}@{REVISION} (T2VA); steps: {args.steps}", flush=True)
    peak = PeakMemory()
    peak.start()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "gpu": gpu, "model": f"{MODEL} (T2VA)", "revision": REVISION,
        "prompt": prompt, "seed": 1101, "width": 1344, "height": 768,
        "frames": 124, "steps": args.steps, "generation_count": 0,
        "transformer_precision": "bf16", "text_encoder_precision": "bf16",
        "attention_backend": "FLASH_ATTN (cuDNN dense implementation)",
    }
    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    model_path = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=["model_index.json", "modular_model_index.json", "scheduler/**", "audio_scheduler/**", "vae/**", "audio_vae/**", "text_encoder/**", "tokenizer/**", "processor/**", "transformer/**"],
                                        local_files_only=True))
    if not (model_path / "model_index.json").exists():
        raise FileNotFoundError(model_path / "model_index.json")

    startup_start = time.time()
    generator = VideoGenerator.from_config({
        "model_path": str(model_path),
        "engine": {
            "num_gpus": 1,
            "compile": {"enabled": False, "vae_enabled": True},
            "offload": {"dit": False, "dit_layerwise": False, "text_encoder": False,
                        "image_encoder": False, "vae": False},
        },
        "pipeline": {"experimental": {"attention_backend": "FLASH_ATTN"}},
    })
    summary["startup_s"] = time.time() - startup_start
    summary["loaded_peak_mib"] = peak.take()
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Loaded original H3 in {summary['startup_s']:.1f}s; generating one clip", flush=True)

    peak.take()
    start = time.time()
    summary["generation_count"] = 1
    summary_path.write_text(json.dumps(summary, indent=2))
    result = generator.generate({
        "prompt": prompt,
        "sampling": {"height": 768, "width": 1344, "num_frames": 124,
                     "num_inference_steps": args.steps, "guidance_scale": 1.0,
                     "seed": 1101, "fps": 24},
        "output": {"output_path": str(OUT), "output_video_name": "original-h3",
                   "return_frames": False},
    })
    row = {
        "client_s": time.time() - start, "generation_s": result.generation_time,
        "peak_mib": peak.take(), "reported_peak_mb": result.peak_memory_mb,
        "video_path": result.video_path,
    }
    stages = getattr(result.logging_info, "stages", None) or {}
    row["stages"] = {key: value.get("execution_time") for key, value in stages.items()}
    if result.video_path:
        row.update(describe(OUT, "original-h3", result.video_path))
    summary["run"] = row
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
