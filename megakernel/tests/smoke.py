"""Exercise dispatch, no-ops and instruction-ring reuse on a reference GPU."""
import torch

def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; this test cannot run on macOS/CPU")
    capability = torch.cuda.get_device_capability()
    if capability not in {(9, 0), (10, 0)}:
        raise SystemExit(f"Reference runtime does not support SM{capability[0]}{capability[1]}; SM120 port is pending")
    from mk_h3 import run
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    # Upstream indexes instruction/timing streams by physical SM ID, which
    # need not be dense. Replicate the smoke schedule over 1024 addressable IDs.
    slots, width, timing_width = 7, 32, 128
    instructions = torch.zeros((1024, slots, width), device="cuda", dtype=torch.int32)
    instructions[:, ::2, 0] = 1  # alternate smoke / NoOp; wrap 2-slot ring
    timings = torch.zeros((1024, slots, timing_width), device="cuda", dtype=torch.int32)
    counts = torch.zeros(sm_count, device="cuda", dtype=torch.int32)
    for _ in range(2):
        counts.zero_()
        run(instructions, timings, counts, sm_count, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        assert torch.equal(counts, torch.full_like(counts, 4)), counts.cpu().tolist()
    print(f"PASS: {sm_count} blocks, mixed opcodes, ring reuse, repeat launch")

if __name__ == "__main__":
    main()
