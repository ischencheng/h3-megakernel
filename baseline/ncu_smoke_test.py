"""Run from the repo root: modal run baseline/ncu_smoke_test.py"""

import modal

app = modal.App("ncu-smoke-test")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("cuda-nsight-compute-12-9")
)

CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s\n", cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

__global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main() {
    const int n = 1 << 20;
    float* x = nullptr;
    CHECK(cudaMalloc(&x, n * sizeof(float)));
    CHECK(cudaMemset(x, 0, n * sizeof(float)));
    add_one<<<(n + 255) / 256, 256>>>(x, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float result = 0;
    CHECK(cudaMemcpy(&result, x, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(x));
    printf("Kernel result: %.1f (expected 1.0)\n", result);
    return result == 1.0f ? 0 : 1;
}
"""


@app.function(gpu="L4", image=image, timeout=300, max_containers=1)
def test():
    import subprocess
    from pathlib import Path

    Path("/tmp/ncu_test.cu").write_text(CUDA_SOURCE)
    subprocess.run(
        ["nvcc", "-O2", "-lineinfo", "-arch=sm_89",
         "/tmp/ncu_test.cu", "-o", "/tmp/ncu_test"],
        check=True,
    )
    print("Checking ordinary CUDA execution first...", flush=True)
    subprocess.run(["/tmp/ncu_test"], check=True)
    print("Testing NCU counter access with clock control disabled...", flush=True)
    result = subprocess.run(
        ["ncu", "--clock-control", "none", "--set", "basic",
         "--launch-count", "1", "--force-overwrite",
         "--export", "/tmp/ncu-smoke-test", "/tmp/ncu_test"],
        check=False,
    )
    reports = {}
    if result.returncode == 0:
        for path in Path("/tmp").glob("ncu-smoke-test.ncu-rep*"):
            reports[path.name] = path.read_bytes()
    return result.returncode, reports


@app.local_entrypoint()
def main():
    from pathlib import Path

    code, reports = test.remote()
    print(f"NCU exit code: {code}. Check the remote output for collected metrics or errors.")
    if code:
        raise RuntimeError(f"NCU smoke test failed with exit code {code}")
    if not reports:
        raise RuntimeError("NCU exited successfully but did not produce a report.")
    destination = Path("/tmp/modal-ncu-reports")
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in reports.items():
        path = destination / name
        path.write_bytes(data)
        print(f"Saved Nsight Compute report: {path.resolve()}")
