import torch

from . import config


# writing a buffer bigger than l2 leaves it cold
class L2Flusher:

    def __init__(self, nbytes=config.L2_FLUSH_BYTES, device="cuda"):
        self.buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)

    def flush(self):
        self.buffer.zero_()
