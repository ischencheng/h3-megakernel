// one persistent kernel, tasks queued per token group (owner: dit)
// fc2 tiles wait on the group's counter before reading h

#include <torch/extension.h>

torch::Tensor mlp_fused(torch::Tensor x, torch::Tensor w1, torch::Tensor s1, torch::Tensor alpha1, torch::Tensor sf1,
                        torch::Tensor w2, torch::Tensor s2, torch::Tensor alpha2, torch::Tensor sf2,
                        int64_t group_tokens) {
  TORCH_CHECK(false, "mlp_fused not implemented yet");
  return x;
}
