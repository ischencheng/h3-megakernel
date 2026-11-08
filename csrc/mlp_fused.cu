// Fused version: one persistent kernel, grid = number of SMs.
// Tasks are queued per token group: all FC1 tiles of a group, then its FC2
// tiles. FC1 tiles bump the group's counter (release); FC2 tiles wait for it
// to be full (acquire) before reading h. group_tokens trades h staying in L2
// against how often the weights are re-read.
// Owner: DiT. Fallback: one cooperative launch with a grid sync between FC1 and FC2.

#include <torch/extension.h>

torch::Tensor mlp_fused(torch::Tensor x, torch::Tensor w1, torch::Tensor w2, int64_t group_tokens) {
  TORCH_CHECK(false, "mlp_fused not implemented yet");
  return x;
}
