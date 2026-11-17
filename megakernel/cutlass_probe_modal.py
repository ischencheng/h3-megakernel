"""Run the pinned SM120 CUTLASS projection probe beside the FastVideo cache."""
from pathlib import Path
import os
import subprocess
import sys

import modal

ROOT = Path(__file__).resolve().parent
if modal.is_local():
    sys.path.insert(0, str(ROOT.parent / "baseline"))
    from fastvideo_modal import image as fastvideo_image, volume

    image = (
        fastvideo_image
        .add_local_dir(str(ROOT), "/root/megakernel", ignore=["third_party/**", "build/**", "**/__pycache__/**"])
        .add_local_dir(str(ROOT / "third_party/ThunderKittens/include"), "/opt/tk/include")
        .add_local_dir(str(ROOT / "third_party/CUTLASS/include"), "/opt/cutlass/include")
        .add_local_dir(str(ROOT / "third_party/CUTLASS/tools/util/include"), "/opt/cutlass/tools/util/include")
    )
else:
    image = modal.Image.debian_slim()
    volume = modal.Volume.from_name("h3-megakernel-cache")
app = modal.App("h3mk-cutlass-sm120-probe")


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def inspect():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download("FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer",
                             revision="9d0d0d3ae69365aded83fab5562f00f9f06c12aa",
                             local_files_only=True)
    weights = Path(path) / "transformer/nvfp4_weights.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as reader:
        keys = [key for key in reader.keys() if "transformer_blocks.0.attn.to_q" in key]
        info = [(key, tuple(reader.get_slice(key).get_shape()), str(reader.get_slice(key).get_dtype()))
                for key in keys]
    print("checkpoint", weights)
    print("projection", info)
    return info


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify(capture_run_id: str):
    if not capture_run_id or "/" in capture_run_id or capture_run_id in (".", ".."):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "cutlass_projection"], cwd=root, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-100:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"CUTLASS compile exited {build.returncode}")
    env = {**os.environ, "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{os.environ.get('PYTHONPATH', '')}",
           "FASTVIDEO_H3_VSA_FP4": "0",
           "H3MK_PROJECTION_CAPTURE": f"/cache/runs/{capture_run_id}"}
    subprocess.run([sys.executable, str(root / "tests/projection_parity.py")],
                   cwd=root, env=env, check=True, timeout=600)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify_hazy(capture_run_id: str, build_only: bool = False, qkv: bool = False):
    if not build_only and (not capture_run_id or "/" in capture_run_id or capture_run_id in (".", "..")):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "THUNDERKITTENS_ROOT=/opt/tk", "hazy_projection"],
                           cwd=root, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-150:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"Hazy projection compile exited {build.returncode}")
    if build_only:
        return "Hazy projection compiled for SM120"
    env = {**os.environ, "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{root / 'tests'}:{os.environ.get('PYTHONPATH', '')}",
           "FASTVIDEO_H3_VSA_FP4": "0",
           "H3MK_PROJECTION_CAPTURE": f"/cache/runs/{capture_run_id}"}
    test_name = "hazy_qkv.py" if qkv else "hazy_projection.py"
    subprocess.run([sys.executable, str(root / "tests" / test_name)],
                   cwd=root, env=env, check=True, timeout=600)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=1800)
def verify_attention_boundary(capture_run_id: str):
    if not capture_run_id or "/" in capture_run_id or capture_run_id in (".", ".."):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    env = {**os.environ,
           "H3MK_ATTENTION_CAPTURE": f"/cache/runs/{capture_run_id}"}
    subprocess.run([sys.executable, str(root / "tests/attention_boundary.py")],
                   cwd=root, env=env, check=True, timeout=1200)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify_hazy_qknorm(capture_run_id: str):
    if not capture_run_id or "/" in capture_run_id or capture_run_id in (".", ".."):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "THUNDERKITTENS_ROOT=/opt/tk", "hazy_projection"],
                           cwd=root, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-100:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"Hazy Q/K norm compile exited {build.returncode}")
    env = {**os.environ,
           "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{root / 'tests'}:{os.environ.get('PYTHONPATH', '')}",
           "H3MK_ATTENTION_CAPTURE": f"/cache/runs/{capture_run_id}"}
    subprocess.run([sys.executable, str(root / "tests/hazy_qknorm_rope.py")],
                   cwd=root, env=env, check=True, timeout=1200)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify_projection_attention(capture_run_id: str, sparse: bool = False):
    if not capture_run_id or "/" in capture_run_id or capture_run_id in (".", ".."):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "THUNDERKITTENS_ROOT=/opt/tk", "hazy_projection"],
                           cwd=root, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-100:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"Hazy projection/attention compile exited {build.returncode}")
    capture = f"/cache/runs/{capture_run_id}"
    env = {**os.environ,
           "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{root / 'tests'}:{os.environ.get('PYTHONPATH', '')}",
           "H3MK_ATTENTION_CAPTURE": capture,
           "H3MK_PROJECTION_CAPTURE": capture,
           "FASTVIDEO_H3_VSA_FP4": "0"}
    command = [sys.executable, str(root / "tests/hazy_projection_attention.py")]
    if sparse:
        command.append("--sparse")
    subprocess.run(command,
                   cwd=root, env=env, check=True, timeout=1800)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify_hazy_sparse(first_row: int, row_count: int):
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "THUNDERKITTENS_ROOT=/opt/tk", "hazy_projection"],
                           cwd=root, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-100:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"Hazy sparse-attention compile exited {build.returncode}")
    env = {**os.environ,
           "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{root / 'tests'}:{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, str(root / "tests/hazy_sparse_attention.py"),
                    "--first-row", str(first_row), "--row-count", str(row_count)],
                   cwd=root, env=env, check=True, timeout=3000)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=1800)
def probe_flashinfer_vsa(head: int, query_tile: int, compare_fastvideo: bool):
    root = Path("/root/megakernel")
    env = {**os.environ, "H3MK_PROBE_HEAD": str(head),
           "H3MK_PROBE_QTILE": str(query_tile),
           "H3MK_COMPARE_FASTVIDEO": "1" if compare_fastvideo else "0"}
    subprocess.run([sys.executable, str(root / "tests/flashinfer_vsa_probe.py")],
                   cwd=root, env=env, check=True, timeout=1500)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=1800)
def verify_vsa_scores(capture_run_id: str):
    if not capture_run_id or "/" in capture_run_id or capture_run_id in (".", ".."):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    env = {**os.environ, "H3MK_REFERENCE_CAPTURE": f"/cache/runs/{capture_run_id}"}
    subprocess.run([sys.executable, str(root / "tests/vsa_scores_boundary.py")],
                   cwd=root, env=env, check=True, timeout=1500)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify_hazy_vsa_pool(capture_run_id: str):
    if not capture_run_id or "/" in capture_run_id or capture_run_id in (".", ".."):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "THUNDERKITTENS_ROOT=/opt/tk", "hazy_projection"],
                           cwd=root, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-100:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"Hazy VSA pool compile exited {build.returncode}")
    env = {**os.environ,
           "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{root / 'tests'}:{os.environ.get('PYTHONPATH', '')}",
           "H3MK_REFERENCE_CAPTURE": f"/cache/runs/{capture_run_id}"}
    subprocess.run([sys.executable, str(root / "tests/hazy_vsa_pool.py")],
                   cwd=root, env=env, check=True, timeout=1800)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify_hazy_vsa_scores(capture_run_id: str):
    if not capture_run_id or "/" in capture_run_id or capture_run_id in (".", ".."):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "THUNDERKITTENS_ROOT=/opt/tk", "hazy_projection"],
                           cwd=root, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-100:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"Hazy VSA scores compile exited {build.returncode}")
    env = {**os.environ,
           "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{root / 'tests'}:{os.environ.get('PYTHONPATH', '')}",
           "H3MK_REFERENCE_CAPTURE": f"/cache/runs/{capture_run_id}"}
    subprocess.run([sys.executable, str(root / "tests/hazy_vsa_scores.py")],
                   cwd=root, env=env, check=True, timeout=1800)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify_hazy_vsa_topk(capture_run_id: str):
    if not capture_run_id or "/" in capture_run_id or capture_run_id in (".", ".."):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "THUNDERKITTENS_ROOT=/opt/tk", "hazy_projection"],
                           cwd=root, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-100:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"Hazy VSA top-k compile exited {build.returncode}")
    env = {**os.environ,
           "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{root / 'tests'}:{os.environ.get('PYTHONPATH', '')}",
           "H3MK_REFERENCE_CAPTURE": f"/cache/runs/{capture_run_id}"}
    subprocess.run([sys.executable, str(root / "tests/hazy_vsa_topk.py")],
                   cwd=root, env=env, check=True, timeout=1800)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify_hazy_vsa_dynamic(capture_run_id: str, compress: bool):
    if not capture_run_id or "/" in capture_run_id or capture_run_id in (".", ".."):
        raise ValueError("capture_run_id must be a single run directory name")
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "THUNDERKITTENS_ROOT=/opt/tk", "hazy_projection"],
                           cwd=root, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-100:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"Hazy dynamic VSA compile exited {build.returncode}")
    env = {**os.environ,
           "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{root / 'tests'}:{os.environ.get('PYTHONPATH', '')}",
           "H3MK_REFERENCE_CAPTURE": f"/cache/runs/{capture_run_id}"}
    command = [sys.executable, str(root / "tests/hazy_vsa_dynamic.py")]
    if compress:
        command.append("--compress")
    subprocess.run(command,
                   cwd=root, env=env, check=True, timeout=1800)


@app.local_entrypoint()
def main(mode: str = "inspect", capture_run_id: str = "",
         first_row: int = 0, row_count: int = 64,
         head: int = 0, query_tile: int = 0,
         compare_fastvideo: bool = False):
    if mode == "inspect":
        print(inspect.remote())
    elif mode == "verify":
        verify.remote(capture_run_id)
    elif mode == "hazy":
        verify_hazy.remote(capture_run_id)
    elif mode == "qkv":
        verify_hazy.remote(capture_run_id, qkv=True)
    elif mode == "compile":
        print(verify_hazy.remote("", True))
    elif mode == "attention-boundary":
        verify_attention_boundary.remote(capture_run_id)
    elif mode == "qknorm":
        verify_hazy_qknorm.remote(capture_run_id)
    elif mode == "projection-attention":
        verify_projection_attention.remote(capture_run_id)
    elif mode == "projection-sparse":
        verify_projection_attention.remote(capture_run_id, sparse=True)
    elif mode == "sparse":
        verify_hazy_sparse.remote(first_row, row_count)
    elif mode == "flashinfer-vsa":
        probe_flashinfer_vsa.remote(head, query_tile, compare_fastvideo)
    elif mode == "vsa-scores":
        verify_vsa_scores.remote(capture_run_id)
    elif mode == "hazy-vsa-pool":
        verify_hazy_vsa_pool.remote(capture_run_id)
    elif mode == "hazy-vsa-scores":
        verify_hazy_vsa_scores.remote(capture_run_id)
    elif mode == "hazy-vsa-topk":
        verify_hazy_vsa_topk.remote(capture_run_id)
    elif mode == "hazy-vsa-dynamic":
        verify_hazy_vsa_dynamic.remote(capture_run_id, False)
    elif mode == "hazy-vsa-full":
        verify_hazy_vsa_dynamic.remote(capture_run_id, True)
    else:
        raise ValueError("mode must be inspect, verify, hazy, qkv, compile, attention-boundary, qknorm, "
                         "projection-attention, projection-sparse, sparse, flashinfer-vsa, vsa-scores, "
                         "hazy-vsa-pool, hazy-vsa-scores, hazy-vsa-topk, hazy-vsa-dynamic, "
                         "or hazy-vsa-full")
