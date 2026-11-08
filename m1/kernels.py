"""Build our CUDA extension and list the variants run.py compares.

Each variant is fn(x, w1, w2) -> y with
  x  [M, HIDDEN]  BF16, contiguous
  w1 [2*FFN, HIDDEN] BF16, gate rows first (see weights.py)
  w2 [HIDDEN, FFN] BF16
  y  [M, HIDDEN]  BF16
"""

import functools

import torch
from torch.utils.cpp_extension import load

from . import config, reference

_CSRC = config.REPO_ROOT / "csrc"
_SOURCES = ["bindings.cpp", "mlp_split.cu", "mlp_fused.cu"]


class NotImplementedKernel(RuntimeError):
    pass


@functools.cache
def extension():
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    return load(
        name=f"h3mk_sm{arch}",
        sources=[str(_CSRC / s) for s in _SOURCES],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
        verbose=False,
    )


def _ours(name, *args):
    try:
        return getattr(extension(), name)(*args)
    except RuntimeError as exc:
        if "not implemented" in str(exc):
            raise NotImplementedKernel(name) from exc
        raise


def variants(groups=config.GROUP_TOKENS):
    """name -> (fn(x, w1, w2), capture_as_graph)."""
    table = {
        "cublas": (reference.mlp_cublas, False),
        "cublas_graph": (reference.mlp_cublas, True),
        "split": (lambda x, w1, w2: _ours("mlp_split", x, w1, w2), False),
        "split_graph": (lambda x, w1, w2: _ours("mlp_split", x, w1, w2), True),
    }
    for g in groups:
        table[f"fused_g{g}"] = (lambda x, w1, w2, g=g: _ours("mlp_fused", x, w1, w2, g), False)
    return table
