"""Headless ComfyUI FL2VA, based on the official H3 T2V template (no Turbo)."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "Comfy-Org/MiniMax-H3"
REVISION = "4b53a35191437de05370da548e59aeabe2a5a09a"
COMFY_COMMIT = "157db2ea2f835917c16191147c11e5896eca01a9"
FILES = {
    "diffusion_models": "minimax_h3_fl2va_int8_convrot.safetensors",
    "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}


def model_files():
    return [("vae" if key.endswith("vae") else key, name) for key, name in FILES.items()]


def graph(prompt, seed, width, height, frames, steps, name):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    return {
        "1": node("UNETLoader", unet_name=FILES["diffusion_models"], weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=FILES["text_encoders"], type="minimax", device="default"),
        "3": node("VAELoader", vae_name=FILES["video_vae"]),
        "4": node("VAELoader", vae_name=FILES["audio_vae"]),
        "5": node("MiniMaxH3ImageToVideo", clip=["2", 0], vae=["3", 0], prompt=prompt,
                  width=width, height=height, length=frames),
        "6": node("RandomNoise", noise_seed=seed),
        "7": node("BasicGuider", model=["1", 0], conditioning=["5", 0]),
        "8": node("KSamplerSelect", sampler_name="res_multistep"),
        "9": node("BasicScheduler", model=["1", 0], scheduler="simple", steps=steps, denoise=1.0),
        "10": node("SamplerCustomAdvanced", noise=["6", 0], guider=["7", 0], sampler=["8", 0],
                   sigmas=["9", 0], latent_image=["5", 1]),
        "11": node("VAEDecode", samples=["10", 0], vae=["3", 0]),
        "12": node("VAEDecodeAudio", samples=["10", 0], vae=["4", 0]),
        "13": node("CreateVideo", images=["11", 0], audio=["12", 0], fps=24, bit_depth=8),
        "14": node("SaveVideo", video=["13", 0], filename_prefix=name,
                   **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}),
    }


def api(path, data=None):
    request = urllib.request.Request("http://127.0.0.1:8188/" + path,
                                     data=None if data is None else json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="/root/baseline/prompts.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1101, 1102])
    parser.add_argument("--shapes", default="1344x768")
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20, help="official non-Turbo template default")
    args = parser.parse_args()
    width, height = map(int, args.shapes.split("x"))
    out = Path(os.environ.get("OUT_DIR", "out")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from fastvideo_capacity import PeakMemory

    for folder, name in model_files():
        source = hf_hub_download(REPO, f"{folder}/{name}", revision=REVISION, local_files_only=True)
        target = Path("/ComfyUI/models") / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source)
    peak = PeakMemory()
    peak.start()
    start = time.monotonic()
    # Keep loaders cached. Distinct prompt/seed pairs invalidate the sampler.
    server = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
                               "--disable-auto-launch", "--output-directory", str(out)],
                              cwd="/ComfyUI")
    summary = {"model": REPO, "revision": REVISION, "comfy_commit": COMFY_COMMIT,
               "files": FILES, "settings": vars(args), "runs": []}
    try:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError("ComfyUI exited during startup; see driver.log")
            try:
                api("system_stats")
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        else:
            raise TimeoutError("ComfyUI did not start in 180 seconds")
        summary["startup_s"] = time.monotonic() - start
        summary["system"] = api("system_stats")
        for i, prompt in enumerate(json.loads(Path(args.prompts).read_text())):
            for seed in args.seeds:
                name = f"p{i}-s{seed}"
                workflow = graph(prompt, seed, width, height, args.frames, args.steps, name)
                (out / f"{name}.json").write_text(json.dumps(workflow, indent=2))
                peak.take()
                started = time.monotonic()
                job = api("prompt", {"prompt": workflow})["prompt_id"]
                while time.monotonic() - started < 3600:
                    if server.poll() is not None:
                        raise RuntimeError("ComfyUI exited during generation")
                    result = api(f"history/{job}").get(job)
                    if result:
                        if result["status"]["status_str"] != "success":
                            raise RuntimeError(json.dumps(result["status"]))
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError(f"Generation exceeded one hour: {name}")
                row = {"prompt_id": i, "prompt": prompt, "seed": seed, "cold": not summary["runs"],
                       "client_s": time.monotonic() - started, "peak_mib": peak.take(),
                       "outputs": result["outputs"]}
                summary["runs"].append(row)
                (out / "summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(row), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()
