import os
from pathlib import Path

# dit shapes
HIDDEN = 5376
FFN = 14336
FC1_OUT = 2 * FFN  # rows [0, FFN) up, [FFN, 2*FFN) gate

# 14830 and 37696 are a 5 s clip at 832x480 and 1344x768:
# 37 latent frames * (h/32) * (w/32), plus 200 audio and about 200 text tokens
TOKENS = (256, 1024, 4096, 14830, 37696)

# one group of h is 29, 59 or 117 mb against a 128 mib l2
GROUP_TOKENS = (1024, 2048, 4096)

SEED = 0

# earlier ops in the model evict these weights, so flush l2 before each run
WARMUP = 3
ITERS = 20
L2_FLUSH_BYTES = 256 << 20  # twice the l2

# allowed error, as a multiple of the serving path's error
ERROR_FACTOR = 2.0

# fasth3 v2 for one gpu
HF_REPO = "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer"
HF_REVISION = "9d0d0d3ae69365aded83fab5562f00f9f06c12aa"
HF_FILE = "transformer/nvfp4_weights.safetensors"
FC1_PREFIX = "transformer_blocks.0.ff.fc_in"
FC2_PREFIX = "transformer_blocks.0.ff.fc_out"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("H3MK_DATA", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("H3MK_RESULTS", REPO_ROOT / "results"))
WEIGHTS_FILE = DATA_DIR / "block0_mlp_nvfp4.safetensors"
