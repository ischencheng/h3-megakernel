"""One reproducible H3 video on one B200, with no warmup or repetitions."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from fastvideo_capacity import ENV, PeakMemory, describe

CONSUMER_MODEL = "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer"
CONSUMER_REVISION = "9d0d0d3ae69365aded83fab5562f00f9f06c12aa"
BF16_MODEL = "FastVideo/FastVideo-FastH3-8-Step-V2"
BF16_REVISION_FILE = Path("/cache/models/bf16_v2_revision.txt")
PROMPTS_FILE = Path("/root/baseline/prompts.json")
OUT = Path(os.environ["OUT_DIR"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("consumer", "bf16"))
    args = parser.parse_args()

    prompts = json.loads(PROMPTS_FILE.read_text())
    if len(prompts) != 1 or not isinstance(prompts[0], str) or not prompts[0].strip():
        raise ValueError("prompts.json must contain exactly one nonempty prompt")
    prompt = prompts[0]
    model = CONSUMER_MODEL if args.variant == "consumer" else BF16_MODEL
    revision = CONSUMER_REVISION if args.variant == "consumer" else BF16_REVISION_FILE.read_text().strip()
    env = {**ENV, "FASTVIDEO_H3_VSA_FP4": "0"}
    os.environ.update(env)

    from fastvideo import VideoGenerator
    from huggingface_hub import snapshot_download

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if "B200" not in gpu:
        raise RuntimeError(f"Expected B200, got {gpu}")
    print(f"GPU: {gpu}; variant: {args.variant}; checkpoint: {model}@{revision}", flush=True)

    peak = PeakMemory()
    peak.start()
    summary = {
        "gpu": gpu, "variant": args.variant, "model": model, "revision": revision,
        "prompt": prompt, "seed": 1101, "width": 1344, "height": 768,
        "frames": 124, "num_inference_steps": 9, "generation_count": 0, "env": env,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    engine = {
        "num_gpus": 1,
        "compile": {"enabled": False, "vae_enabled": True},
        "offload": {"dit": False, "dit_layerwise": False, "text_encoder": False,
                    "image_encoder": False, "vae": False},
    }
    if args.variant == "consumer":
        engine["quantization"] = {"transformer_quant": "NVFP4", "layer_profile": "h3_dit_vsa"}
    startup_start = time.time()
    generator = VideoGenerator.from_config({
        "model_path": snapshot_download(model, revision=revision, local_files_only=True),
        "engine": engine,
        "pipeline": {"experimental": {"attention_backend": "VIDEO_SPARSE_ATTN_H3",
                                      "VSA_sparsity": 0.8, "VSA_tile_size": 64}},
    })
    summary["startup_s"] = time.time() - startup_start
    summary["loaded_peak_mib"] = peak.take()
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Loaded in {summary['startup_s']:.1f}s; now generating the only clip", flush=True)

    peak.take()
    generation_start = time.time()
    summary["generation_count"] = 1
    summary_path.write_text(json.dumps(summary, indent=2))
    result = generator.generate({
        "prompt": prompt,
        "sampling": {"height": 768, "width": 1344, "num_frames": 124,
                     "num_inference_steps": 9, "guidance_scale": 1.0,
                     "seed": 1101, "fps": 24},
        "output": {"output_path": str(OUT), "output_video_name": args.variant,
                   "return_frames": False},
    })
    row = {
        "client_s": time.time() - generation_start,
        "generation_s": result.generation_time,
        "peak_mib": peak.take(),
        "reported_peak_mb": result.peak_memory_mb,
        "video_path": result.video_path,
    }
    stages = getattr(result.logging_info, "stages", None) or {}
    row["stages"] = {key: value.get("execution_time") for key, value in stages.items()}
    if result.video_path:
        row.update(describe(OUT, args.variant, result.video_path))
    summary["run"] = row
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
