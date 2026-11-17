# the h3 video decoder block as eight kernels (kernels.cu) and fastvideo's attention
import functools
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
HIDDEN, HEADS, HEAD_DIM, ROTARY = 2048, 32, 64, 48


def _cutlass_include():
    root = Path(os.environ.get("H3MK_CUTLASS", "/FastVideo/fastvideo-kernel/include/cutlass"))
    return [str(root / "include"), str(root / "tools" / "util" / "include")]


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="h3mk_video_decoder",
        sources=[str(HERE / "kernels.cu")],
        extra_include_paths=_cutlass_include(),
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_120a,code=sm_120a",
                           "-DNDEBUG", "-Xptxas=-v"],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=_build_dir(),
        verbose=os.environ.get("H3MK_EXT_VERBOSE") == "1",  # prints ptxas register and spill counts
    )


def _build_dir():
    # one folder per source version: ninja does not see a mounted kernels.cu as newer than the cached build
    digest = hashlib.sha1((HERE / "kernels.cu").read_bytes()).hexdigest()[:12]
    path = Path(os.environ.get("H3MK_EXT_DIR", "/tmp/h3mk-ext")) / f"video_decoder-{digest}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# one block's weights in the layout the kernels read
class Packed:

    def __init__(self, block):
        attn, ff = block.attn, block.ff

        def lin(l):
            bias = l.bias.float() if l.bias is not None else torch.zeros(l.out_features, device=l.weight.device)
            return l.weight, l.weight_scale.reshape(-1).float(), bias

        q, k, v = (lin(l) for l in (attn.to_q, attn.to_k, attn.to_v))
        self.w_qkv = torch.cat([q[0], k[0], v[0]]).contiguous()
        self.s_qkv = torch.cat([q[1], k[1], v[1]]).contiguous()
        self.b_qkv = torch.cat([q[2], k[2], v[2]]).contiguous()
        self.w_out, self.s_out, self.b_out = (t.contiguous() for t in lin(attn.to_out[0]))
        # interleave the value rows [0, F) and gate rows [F, 2F), so one thread holds both halves of a channel
        w1, s1, b1 = lin(ff.net[0].proj)
        f = w1.shape[0] // 2
        order = torch.stack([torch.arange(f), torch.arange(f) + f], dim=1).reshape(-1).to(w1.device)
        self.w1, self.s1, self.b1 = (t.index_select(0, order).contiguous() for t in (w1, s1, b1))
        self.w2, self.s2, self.b2 = (t.contiguous() for t in lin(ff.net[2]))
        self.norm1, self.norm2 = block.norm1.weight.float().contiguous(), block.norm2.weight.float().contiguous()
        self.eps1, self.eps2 = block.norm1.eps, block.norm2.eps
        self.qk_eps = attn.norm_q.eps
        self.gate1, self.gate2 = block.scale1.float().contiguous(), block.scale2.float().contiguous()
        for l in (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0], ff.net[0].proj, ff.net[2]):
            assert l.convrot and l.group_size == 256, "the rotation kernel is the 256 group ConvRot"


# x is [B, S, 2048]; inplace updates a contiguous fp32 x
@torch.no_grad()
def forward(block, packed, x, rotary_emb, inplace=False):
    e = ext()
    shape = x.shape
    m = shape[0] * shape[1]
    xf = x.reshape(m, HIDDEN)
    if xf.dtype != torch.float32 or not xf.is_contiguous():
        xf = xf.float().contiguous()
    elif not inplace:
        xf = xf.clone()
    cos, sin = (t.reshape(m, ROTARY).float().contiguous() for t in rotary_emb)
    a, s = e.rowquant(xf, packed.norm1, packed.eps1)
    q, k, v = e.gemm_qkv(a, s, packed.w_qkv, packed.s_qkv, packed.b_qkv, cos, sin, packed.qk_eps)
    heads = lambda t: t.view(shape[0], shape[1], HEADS, HEAD_DIM)
    o = block.attn.attn_impl.forward(heads(q), heads(k), heads(v), None)
    o = o.reshape(m, HIDDEN).contiguous()
    a, s = e.rowquant(o, None, 0.0)
    e.gemm_residual(a, s, packed.w_out, packed.s_out, packed.b_out, packed.gate1, xf)
    a, s = e.rowquant(xf, packed.norm2, packed.eps2)
    h = e.gemm_swiglu(a, s, packed.w1, packed.s1, packed.b1)
    a, s = e.rowquant(h, None, 0.0)
    e.gemm_residual(a, s, packed.w2, packed.s2, packed.b2, packed.gate2, xf)
    return xf.view(shape[0], shape[1], HIDDEN)


# fp32 reference: dequantized weights, exact rotation, no activation rounding

def _hadamard(device):
    from fastvideo.models.vaes.minimax_h3_int8_convrot import regular_hadamard

    return regular_hadamard(256, device=device, dtype=torch.float32)


def _linear_fp32(layer, x, hadamard):
    rows = x.reshape(-1, x.shape[-1])
    rotated = (rows.reshape(rows.shape[0], -1, 256) @ hadamard).reshape(rows.shape)
    w = layer.weight.float() * layer.weight_scale.float()
    return (rotated @ w.T + layer.bias.float()).reshape(*x.shape[:-1], w.shape[0])


# a few tiles at a time to bound the attention memory
@torch.no_grad()
def reference_fp32(block, x, rotary_emb, tiles_per_chunk=4):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type="cuda", enabled=False):  # the decode stage runs under fp16 autocast
            return _reference_fp32(block, x, rotary_emb, tiles_per_chunk)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _reference_fp32(block, x, rotary_emb, tiles_per_chunk):
    from fastvideo.models.vaes.minimax_h3_video import _apply_rotary

    hadamard = _hadamard(x.device)
    attn, ff = block.attn, block.ff
    outs = []
    for start in range(0, x.shape[0], tiles_per_chunk):
        h = x[start:start + tiles_per_chunk].float()
        cos, sin = (t[start:start + tiles_per_chunk].float() for t in rotary_emb)
        n = F.rms_norm(h, (HIDDEN,), block.norm1.weight.float(), block.norm1.eps)
        q, k, v = (_linear_fp32(l, n, hadamard).unflatten(2, (HEADS, -1)) for l in (attn.to_q, attn.to_k, attn.to_v))
        q = F.rms_norm(q, (HEAD_DIM,), eps=attn.norm_q.eps)
        k = F.rms_norm(k, (HEAD_DIM,), eps=attn.norm_k.eps)
        q, k = _apply_rotary(q, k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           scale=HEAD_DIM**-0.5).transpose(1, 2).flatten(2, 3)
        h = h + _linear_fp32(attn.to_out[0], o, hadamard) * block.scale1.float()
        n = F.rms_norm(h, (HIDDEN,), block.norm2.weight.float(), block.norm2.eps)
        u, g = _linear_fp32(ff.net[0].proj, n, hadamard).chunk(2, dim=-1)
        h = h + _linear_fp32(ff.net[2], u * F.silu(g), hadamard) * block.scale2.float()
        outs.append(h)
    return torch.cat(outs)
