import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "out"))
MODEL = "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer"
REVISION = "9d0d0d3ae69365aded83fab5562f00f9f06c12aa"

PROMPT = "A red fox runs through fresh snow at dawn, with pawsteps, winter wind and distant birds."


class PeakMemory(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.lock = threading.Lock()

    def run(self):
        while True:
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()
            if used:
                with self.lock:
                    self.peak = max(self.peak, int(used[0]))
            time.sleep(0.5)

    # peak since the last call
    def take(self):
        with self.lock:
            peak, self.peak = self.peak, 0
        return peak


ENV = {
    "FASTVIDEO_H3_VSA_FP4": "1",            # sparse fp4 attention
    "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",  # fused norm, modulate, rope and swiglu
    "FASTVIDEO_NVFP4_MM_BACKEND": "cutlass",
    "FASTVIDEO_H3_VAE_TILE_BATCH": "28",    # one decoder call per 1344x768 tile grid
    "FASTVIDEO_STAGE_LOGGING": "1",
}


def describe(folder, name, video_path):
    row = {}
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "2", "-i", video_path, "-frames:v", "1",
                    str(folder / f"{name}.png")])
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_name,width,height,nb_frames,sample_rate,channels", "-of", "json", video_path],
                           capture_output=True, text=True)
    row["streams"] = json.loads(probe.stdout or "{}").get("streams")
    return row


def main():
    parser = argparse.ArgumentParser(description="record the peak memory and time of fasth3 v2 on one gpu")
    parser.add_argument("--shapes", nargs="+", default=["1344x768"], help="one per process")
    parser.add_argument("--frames", type=int, default=124, help="17n+5; 124 is 5 s")
    parser.add_argument("--reps", type=int, default=3, help="the first run at each shape is a warmup")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(ENV)
    peak = PeakMemory()
    peak.start()

    from fastvideo import VideoGenerator
    from huggingface_hub import snapshot_download

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("GPU:", gpu, flush=True)
    summary = {"gpu": gpu, "model": MODEL, "revision": REVISION, "env": ENV, "runs": []}
    path = OUT / "summary.json"

    start = time.time()
    generator = VideoGenerator.from_config({
        "model_path": snapshot_download(MODEL, revision=REVISION, local_files_only=True),
        "engine": {
            "num_gpus": 1,
            "quantization": {"transformer_quant": "NVFP4", "layer_profile": "h3_dit_vsa"},
            "compile": {"enabled": False, "vae_enabled": True},
            # a megakernel needs every component on the gpu
            "offload": {"dit": False, "dit_layerwise": False, "text_encoder": False, "image_encoder": False, "vae": False},
        },
        "pipeline": {"experimental": {"attention_backend": "VIDEO_SPARSE_ATTN_H3", "VSA_sparsity": 0.8,
                                      "VSA_tile_size": 64}},
    })
    summary.update(startup_s=time.time() - start, loaded_peak_mib=peak.take())
    time.sleep(5)
    peak.take()
    time.sleep(5)
    summary["resident_mib"] = peak.take()
    print(f"loaded in {summary['startup_s']:.0f}s, peak {summary['loaded_peak_mib']} MiB, "
          f"resident {summary['resident_mib']} MiB", flush=True)
    path.write_text(json.dumps(summary, indent=1))

    for shape in args.shapes:
        w, h = map(int, shape.split("x"))
        for rep in range(args.reps):
            name = f"{shape}-r{rep}"
            peak.take()
            start = time.time()
            result = generator.generate({
                "prompt": PROMPT,
                "sampling": {"height": h, "width": w, "num_frames": args.frames, "num_inference_steps": 9,
                             "guidance_scale": 1.0, "seed": 1101, "fps": 24},
                "output": {"output_path": str(OUT), "output_video_name": name, "return_frames": False},
            })
            row = {"shape": shape, "rep": rep, "client_s": time.time() - start, "peak_mib": peak.take(),
                   "generation_s": result.generation_time, "reported_peak_mb": result.peak_memory_mb,
                   "video_path": result.video_path}
            stages = getattr(result.logging_info, "stages", None) or {}
            row["stages"] = {key: value.get("execution_time") for key, value in stages.items()}
            if result.video_path:
                row.update(describe(OUT, name, result.video_path))
            summary["runs"].append(row)
            print(json.dumps(row), flush=True)
            path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
