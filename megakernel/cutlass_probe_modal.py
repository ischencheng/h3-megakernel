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


def _nsight(name: str) -> str:
    import glob
    import shutil

    return shutil.which(name) or sorted(glob.glob(f"/opt/nvidia/nsight-*/*/{name}") +
                                        glob.glob(f"/opt/nvidia/nsight-*/*/bin/{name}"))[-1]


def _run_dit(script: str, argv: list[str], timeout: int, profiler: str = "", ncu_args: str = "",
             timings: bool = False):
    """Build the full-DiT Hazy module and run one of its gates beside FastVideo.

    `profiler` wraps the gate in Nsight Systems ("nsys") or Nsight Compute
    ("ncu"). Both record only between cudaProfilerStart and Stop, which the
    gate places with --profile-range. Reports go to `<--out>-profile/`.
    """
    import shlex

    if "/" in script or not script.endswith(".py"):
        raise ValueError("script must name a file in megakernel/tests")
    if profiler not in ("", "nsys", "ncu"):
        raise ValueError("profiler must be nsys or ncu")
    root = Path("/root/megakernel")
    subprocess.run(["uv", "pip", "install", "--system", "pybind11"], check=True)
    build = subprocess.run(["make", "GPU=RTXPRO6000", "CUTLASS_ROOT=/opt/cutlass",
                            "THUNDERKITTENS_ROOT=/opt/tk", f"TIMINGS={int(timings)}", "hazy_dit"],
                           cwd=root, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print("\n".join(build.stdout.splitlines()[-150:]), flush=True)
    if build.returncode:
        raise RuntimeError(f"Hazy DiT compile exited {build.returncode}")
    env = {**os.environ,
           "PYTHONPATH": f"{root / 'build/RTXPRO6000'}:{root}:{root / 'tests'}:/root/baseline:"
                         f"{os.environ.get('PYTHONPATH', '')}",
           "PYTHONUNBUFFERED": "1", "FASTVIDEO_H3_VSA_FP4": "0"}
    command = [sys.executable, str(root / "tests" / script), *argv]
    if timings and "--record-timings" not in command:
        command.append("--record-timings")
    profile_dir = None
    if profiler:
        profile_dir = Path(argv[argv.index("--out") + 1] + "-profile")
        profile_dir.mkdir(parents=True, exist_ok=False)
        if profiler == "nsys":
            # gVisor has no perf_event_open, so no CPU sampling.
            command = [_nsight("nsys"), "profile", "--trace=cuda,nvtx", "--sample=none",
                       "--cpuctxsw=none", "--cuda-graph-trace=node",
                       "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                       "--output", str(profile_dir / "request"), *command]
        else:
            # Modal cannot lock the GPU clocks.
            command = [_nsight("ncu"), "--clock-control", "none", "--profile-from-start", "off",
                       "--export", str(profile_dir / "kernels"), *shlex.split(ncu_args), *command]
    returncode = -1
    try:
        with open((profile_dir or Path("/tmp")) / "driver.log", "w") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait(timeout=timeout)
        if profiler == "nsys" and (profile_dir / "request.nsys-rep").exists():
            subprocess.run([_nsight("nsys"), "stats", "--report",
                            "nvtx_gpu_proj_sum,cuda_gpu_kern_sum", "--format", "csv", "--output",
                            str(profile_dir / "request"), str(profile_dir / "request.nsys-rep")])
        if profiler == "ncu" and (profile_dir / "kernels.ncu-rep").exists():
            page = subprocess.run([_nsight("ncu"), "--import", str(profile_dir / "kernels.ncu-rep"),
                                   "--page", "details"], text=True, capture_output=True)
            (profile_dir / "kernels-details.txt").write_text(page.stdout + page.stderr)
    finally:
        if profile_dir is not None:
            (profile_dir / "returncode").write_text(str(returncode))
        volume.commit()
    if returncode:
        raise RuntimeError(f"{script} exited {returncode}")


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              volumes={"/cache": volume}, timeout=3600)
def verify_dit_ops(argv: list[str], script: str = "dit_ops.py"):
    _run_dit(script, argv, 3000)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=(8, 32), memory=(131072, 393216),
              volumes={"/cache": volume}, timeout=10800)
def verify_dit(script: str, argv: list[str], profiler: str = "", ncu_args: str = "",
               timings: bool = False):
    _run_dit(script, argv, 10200, profiler, ncu_args, timings)


@app.local_entrypoint()
def main(mode: str = "inspect", capture_run_id: str = "",
         first_row: int = 0, row_count: int = 64,
         head: int = 0, query_tile: int = 0,
         compare_fastvideo: bool = False, script: str = "hazy_dit.py",
         args: str = "", profiler: str = "", ncu_args: str = "", spawn: bool = False,
         timings: bool = False):
    import shlex

    if mode == "dit-ops":
        verify_dit_ops.remote(shlex.split(args), script if script != "hazy_dit.py" else "dit_ops.py")
    elif mode == "dit":
        if spawn:
            # With `modal run --detach` the call outlives this client.
            call = verify_dit.spawn(script, shlex.split(args), profiler, ncu_args, timings)
            print("spawned", call.object_id)
        else:
            verify_dit.remote(script, shlex.split(args), profiler, ncu_args, timings)
    elif mode == "inspect":
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
                         "hazy-vsa-full, dit-ops, or dit")
