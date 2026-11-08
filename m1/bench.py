import statistics

import torch

from . import config


# writing a buffer bigger than l2 leaves it cold
class L2Flusher:

    def __init__(self, nbytes=config.L2_FLUSH_BYTES, device="cuda"):
        self.buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)

    def flush(self):
        self.buffer.zero_()


def capture_graph(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def time_fn(fn, name, warmup=config.WARMUP, iters=config.ITERS, flush=True):
    flusher = L2Flusher() if flush else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    events = []
    for _ in range(iters):
        if flusher is not None:
            flusher.flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(name)
        start.record()
        fn()
        end.record()
        torch.cuda.nvtx.range_pop()
        events.append((start, end))
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in events)
    q = statistics.quantiles(times, n=10)
    median = statistics.median(times)
    return {
        "median_ms": median,
        "p10_ms": q[0],
        "p90_ms": q[-1],
        "min_ms": times[0],
        "max_ms": times[-1],
        "spread": (q[-1] - q[0]) / median,
        "iters": iters,
        "l2_flush": flush,
    }
