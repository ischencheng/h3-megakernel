"""Inputs and the two reference implementations of the MLP.

mlp_fp32 is the ground truth: FP32 math with TF32 off.
mlp_cublas is what PyTorch does in BF16 (cuBLAS GEMMs). Its error against
mlp_fp32 sets the bar our kernels have to meet, and its time is the context
for our numbers.
"""

import torch
import torch.nn.functional as F

from . import config


def make_input(tokens, device="cuda"):
    """Random activations at the scale of the norm2 + AdaLN output.

    Milestone 1 has no captured activations yet. The MLP works token by token,
    so random rows test the same math; real activations come with the
    vLLM-Omni baseline in Milestone 2.
    """
    gen = torch.Generator(device=device).manual_seed(config.SEED + tokens)
    x = torch.randn(tokens, config.HIDDEN, generator=gen, device=device, dtype=torch.float32)
    return x.to(torch.bfloat16)


def swiglu(h):
    """silu(gate) * up on a [M, 2*FFN] tensor laid out gate-first."""
    gate, up = h.chunk(2, dim=-1)
    return F.silu(gate) * up


@torch.no_grad()
def mlp_fp32(x, w1, w2, rows_per_chunk=4096):
    """FP32 reference, computed in token chunks to bound memory."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        w1f, w2f = w1.float(), w2.float()
        out = torch.empty(x.shape[0], config.HIDDEN, device=x.device, dtype=torch.float32)
        for start in range(0, x.shape[0], rows_per_chunk):
            xs = x[start:start + rows_per_chunk].float()
            out[start:start + rows_per_chunk] = swiglu(xs @ w1f.T) @ w2f.T
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@torch.compile
def _swiglu_bf16(h):
    # One fused kernel, like vLLM-Omni's SiluAndMul: FP32 math, one BF16 rounding.
    # Written as plain PyTorch it would be five elementwise kernels.
    gate, up = h.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


@torch.no_grad()
def mlp_cublas(x, w1, w2):
    """BF16 path: cuBLAS GEMM, fused SwiGLU, cuBLAS GEMM."""
    return F.linear(_swiglu_bf16(F.linear(x, w1)), w2)
