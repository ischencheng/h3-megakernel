import torch
import torch.nn.functional as F

from . import config

# e2m1 by nibble, bit 3 is the sign
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


# up first, then gate
def swiglu(h):
    up, gate = h.chunk(2, dim=-1)
    return up * F.silu(gate)


# fp32 [out, in] weights from the packed bytes
def dequantize(layer):
    rows, columns = layer.w.shape[0], layer.w.shape[1] * 2
    lut = torch.tensor(_E2M1, device=layer.w.device, dtype=torch.float32)
    # a uint8 index would be a boolean mask
    values = torch.stack((lut[(layer.w & 0xF).int()], lut[(layer.w >> 4).int()]), dim=-1).reshape(rows, columns)
    # undo the 128x4 swizzle
    linear = layer.s.reshape(rows // 128, columns // 64, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, columns // 16)
    scales = linear.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return values * scales * layer.alpha


@torch.no_grad()
def mlp_fp32(x, w, rows_per_chunk=4096):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        w1f, w2f = dequantize(w.fc1), dequantize(w.fc2)
        out = torch.empty(x.shape[0], config.HIDDEN, device=x.device, dtype=torch.float32)
        for start in range(0, x.shape[0], rows_per_chunk):
            xs = x[start:start + rows_per_chunk].float()
            out[start:start + rows_per_chunk] = swiglu(xs @ w1f.T) @ w2f.T
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
