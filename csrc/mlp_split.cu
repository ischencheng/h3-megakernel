// two launches: fc1 + swiglu, then fc2 (owner: encoder)
// x and h are quantized to nvfp4 inside the kernel

#include <torch/extension.h>

torch::Tensor mlp_split(torch::Tensor x, torch::Tensor w1, torch::Tensor s1, torch::Tensor alpha1, torch::Tensor sf1,
                        torch::Tensor w2, torch::Tensor s2, torch::Tensor alpha2, torch::Tensor sf2) {
  TORCH_CHECK(false, "mlp_split not implemented yet");
  return x;
}
