"""Shared settings for Milestone 1: the MLP of one MiniMax-H3 DiT block.

Every script reads its shapes, token counts and measurement rules from here,
so the three of us always compare the same thing.
"""

import os
from pathlib import Path

# DiT shapes (transformer/config.json: hidden_size 5376, ffn_dim 14336).
HIDDEN = 5376
FFN = 14336
FC1_OUT = 2 * FFN  # fc1 holds gate and up stacked: rows [0, FFN) gate, [FFN, 2*FFN) up

# Token counts to sweep. 14,830 and 37,696 are the DiT sequence lengths of a
# 5 s T2VA request at 832x480 and 1344x768. 5 s is 124 frames (H3 snaps to
# 17n+5), which the VAE maps to 5n+2 = 37 latent frames; the DiT patch is
# 1x2x2 on a /16 latent, so 37 * (H/32) * (W/32) = 14,430 or 37,296 video
# tokens, plus 200 audio latents (40 Hz) and about 200 text tokens (prompt
# dependent, assumed). See vllm_omni/diffusion/models/minimax_h3/time_request.py.
TOKENS = (256, 1024, 4096, 14830, 37696)

# Token-group sizes for the fused kernel. A group's intermediate h is
# group * FFN * 2 bytes: 29 MB, 59 MB and 117 MB against a 128 MiB L2.
GROUP_TOKENS = (1024, 2048, 4096)

SEED = 0

# Timing: median of ITERS runs after WARMUP runs, with L2 flushed before each
# run because earlier ops in the real model evict these weights.
WARMUP = 3
ITERS = 20
L2_FLUSH_BYTES = 256 << 20  # twice the 128 MiB L2 of the RTX PRO 6000

# A kernel passes if its error against the FP32 reference is at most
# ERROR_FACTOR times the error of the cuBLAS BF16 path on the same input.
ERROR_FACTOR = 2.0

# Block-0 MLP weights, read straight from the pinned base checkpoint.
HF_REPO = "MiniMaxAI/MiniMax-H3"
HF_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
HF_SHARD = "transformer/diffusion_pytorch_model-00001-of-00014.safetensors"
FC1_KEY = "transformer_blocks.0.ff.net.0.proj.weight"
FC2_KEY = "transformer_blocks.0.ff.net.2.weight"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("H3MK_DATA", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("H3MK_RESULTS", REPO_ROOT / "results"))
WEIGHTS_FILE = DATA_DIR / "block0_mlp.safetensors"
