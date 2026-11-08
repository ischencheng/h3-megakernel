#include <torch/extension.h>

// Both take x [M, 5376], w1 [28672, 5376] (gate rows first), w2 [5376, 14336],
// all BF16 and contiguous, and return y [M, 5376] in BF16.
torch::Tensor mlp_split(torch::Tensor x, torch::Tensor w1, torch::Tensor w2);
torch::Tensor mlp_fused(torch::Tensor x, torch::Tensor w1, torch::Tensor w2, int64_t group_tokens);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mlp_split", &mlp_split, "FC1 + SwiGLU kernel, then FC2 kernel (shared GEMM tile)");
  m.def("mlp_fused", &mlp_fused, "FC1, SwiGLU and FC2 in one persistent kernel");
}
