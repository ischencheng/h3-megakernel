"""Exercise dispatch, page release, ring reuse and physical-SM scheduling."""
import torch

def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; this test cannot run on macOS/CPU")
    capability = torch.cuda.get_device_capability()
    if capability not in {(9, 0), (10, 0), (12, 0)}:
        raise SystemExit(f"Runtime does not support SM{capability[0]}{capability[1]}")
    from mk_h3 import run
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    # Upstream indexes instruction/timing streams by physical SM ID, which
    # need not be dense. Replicate the smoke schedule over 1024 addressable IDs.
    slots, width, timing_width = 9, 32, 128
    instructions = torch.zeros((1024, slots, width), device="cuda", dtype=torch.int32)
    instructions[:, ::2, 0] = 1  # alternate smoke / NoOp; wrap 2-slot ring
    instructions[:, 7, 0] = 2  # all SMs must reach this instruction
    timings = torch.zeros((1024, slots, timing_width), device="cuda", dtype=torch.int32)
    counts = torch.zeros(sm_count, device="cuda", dtype=torch.int32)
    physical_sm_ids = torch.full((sm_count,), -1, device="cuda", dtype=torch.int32)
    rendezvous_count = torch.zeros(1, device="cuda", dtype=torch.int32)
    graph_arena = torch.zeros(1, device="cuda", dtype=torch.float32)
    wave_counts = torch.zeros(1, device="cuda", dtype=torch.int32)
    graph_error = torch.zeros(1, device="cuda", dtype=torch.int32)
    for _ in range(2):
        counts.zero_()
        physical_sm_ids.fill_(-1)
        rendezvous_count.zero_()
        run(instructions, timings, counts, physical_sm_ids, rendezvous_count,
            graph_arena, wave_counts, graph_error, sm_count,
            stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        assert torch.equal(counts, torch.full_like(counts, 5)), counts.cpu().tolist()
        assert rendezvous_count.item() == sm_count
        ids = physical_sm_ids.cpu().tolist()
        assert len(set(ids)) == sm_count and min(ids) >= 0 and max(ids) < 1024, ids
    print(f"PASS: {sm_count} blocks, mixed opcodes, ring reuse, cross-SM rendezvous, repeat launch")

if __name__ == "__main__":
    main()
