#include <torch/extension.h>

// x is [M, 5376] bf16; each linear has packed fp4 w, swizzled fp8 s, alpha and sf
torch::Tensor mlp_split(torch::Tensor x, torch::Tensor w1, torch::Tensor s1, torch::Tensor alpha1, torch::Tensor sf1,
                        torch::Tensor w2, torch::Tensor s2, torch::Tensor alpha2, torch::Tensor sf2);
torch::Tensor mlp_fused(torch::Tensor x, torch::Tensor w1, torch::Tensor s1, torch::Tensor alpha1, torch::Tensor sf1,
                        torch::Tensor w2, torch::Tensor s2, torch::Tensor alpha2, torch::Tensor sf2, int64_t group_tokens);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mlp_split", &mlp_split, "FC1 + SwiGLU kernel, then FC2 kernel (shared GEMM tile)");
  m.def("mlp_fused", &mlp_fused, "FC1, SwiGLU and FC2 in one persistent kernel");
}
