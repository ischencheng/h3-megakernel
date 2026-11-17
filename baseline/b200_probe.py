"""Short scheduling diagnostic; does not load a model or generate media."""

import subprocess

import modal

app = modal.App("h3mk-b200-capacity-probe")


@app.function(gpu="B200", retries=0, timeout=60)
def probe():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    ).strip()


@app.local_entrypoint()
def go():
    print(probe.remote())
