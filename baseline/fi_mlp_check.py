"""Quick check of the one-launch MLP block: build it and compare with FlashInfer's three launches, bitwise."""
import json
import sys
import time

import torch

from fi_mlp_bench import EPS, inputs
from harness import weights
from stages.dit import fi_mlp


def main():
    w = weights.load("cuda")
    p = fi_mlp.pack(w.fc1, w.fc2)
    t0 = time.time()
    module = fi_mlp.build()
    print(f"built in {time.time() - t0:.0f} s", flush=True)
    for m in [int(t) for t in sys.argv[1:]] or [256, 1024, 37730]:
        a = inputs(m)
        ref = fi_mlp.run(*a, a[0], p, EPS)
        out = fi_mlp.run_one_launch(*a, a[0], p, EPS, module=module)
        torch.cuda.synchronize()
        print(json.dumps({"tokens": m, "equal": torch.equal(out, ref), "mismatched": int((out != ref).sum())}),
              flush=True)


if __name__ == "__main__":
    main()
