// Split baseline: two launches that share the GEMM tile with mlp_fused.cu.
//   kernel 1: h = silu(x @ w1[:FFN].T) * (x @ w1[FFN:].T), written once as BF16
//   kernel 2: y = h @ w2.T
// Owner: encoder. Accumulate in FP32; the checks in m1/check.py decide pass/fail.

#include <torch/extension.h>

torch::Tensor mlp_split(torch::Tensor x, torch::Tensor w1, torch::Tensor w2) {
  TORCH_CHECK(false, "mlp_split not implemented yet");
  return x;
}
