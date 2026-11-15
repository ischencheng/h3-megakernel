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
    parser.add_argument("--fp4-attention", type=int, choices=(0, 1), default=1)
    parser.add_argument("--prompts", help="JSON list of prompt strings; overrides --reps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1101])
    parser.add_argument("--profile-rep", type=int, default=None,
                        help="only this rep runs inside cudaProfilerStart/Stop, for nsys or ncu")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    env = {**ENV, "FASTVIDEO_H3_VSA_FP4": str(args.fp4_attention)}
    os.environ.update(env)
    prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else [PROMPT]
    cases = [(i, prompt, seed) for i, prompt in enumerate(prompts) for seed in args.seeds]
    if not args.prompts:
        cases *= args.reps
    if args.profile_rep is not None:
        os.environ["FASTVIDEO_NVTX_PROFILE"] = "1"  # stage and block ranges
    peak = PeakMemory()
    peak.start()

    import torch
    from fastvideo import VideoGenerator
    from huggingface_hub import snapshot_download

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("GPU:", gpu, flush=True)
    summary = {"gpu": gpu, "model": MODEL, "revision": REVISION, "env": env, "runs": []}
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
        for rep, (prompt_id, prompt, seed) in enumerate(cases):
            name = f"{shape}-r{rep}"
            profile = rep == args.profile_rep
            peak.take()
            if profile:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
            start = time.time()
            result = generator.generate({
                "prompt": prompt,
                "sampling": {"height": h, "width": w, "num_frames": args.frames, "num_inference_steps": 9,
                             "guidance_scale": 1.0, "seed": seed, "fps": 24},
                "output": {"output_path": str(OUT), "output_video_name": name, "return_frames": False},
            })
            if profile:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            row = {"shape": shape, "rep": rep, "prompt_id": prompt_id, "prompt": prompt, "seed": seed,
                   "cold": rep == 0, "client_s": time.time() - start, "peak_mib": peak.take(),
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
