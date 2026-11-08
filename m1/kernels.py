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


def _split(x, w):
    return _ours("mlp_split", x, *w.fc1, *w.fc2)


# name -> (fn(x, w), capture_as_graph)
def variants(groups=config.GROUP_TOKENS):
    table = {
        "flashinfer": (reference.mlp_flashinfer, False),
        "flashinfer_graph": (reference.mlp_flashinfer, True),
        "split": (_split, False),
        "split_graph": (_split, True),
    }
    for g in groups:
        table[f"fused_g{g}"] = (lambda x, w, g=g: _ours("mlp_fused", x, *w.fc1, *w.fc2, g), False)
    return table
