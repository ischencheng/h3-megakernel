"""Error metrics against the FP32 reference and the pass rule."""

import math

from . import config


def error_stats(y, ref):
    """rel_l2 = ||y - ref|| / ||ref||; max_over_rms = max|y - ref| / rms(ref)."""
    diff = y.float() - ref
    ref_norm = ref.norm().item()
    rms = ref.pow(2).mean().sqrt().item()
    return {
        "rel_l2": diff.norm().item() / ref_norm,
        "max_over_rms": diff.abs().max().item() / rms,
    }


def passes(stats, bar, factor=config.ERROR_FACTOR):
    """True if every metric is finite and at most `factor` times the cuBLAS bar."""
    for key, value in stats.items():
        if not math.isfinite(value) or value > factor * bar[key]:
            return False
    return True
