"""Synthetic multi-op, cross-SM, arena-reuse check on RTX PRO 6000."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiler import Graph, Node, Operation, Resources, compile_graph


def fixture():
    return Graph(
        inputs={"x": 4097},
        nodes=(
            Node("a", "affine", ("x",), 4097, scale=2, bias=1),
            Node("b", "affine", ("a",), 4097, scale=3, bias=-2),
            Node("c", "add", ("a", "b"), 4097),
            Node("d", "affine", ("c",), 4097, scale=0.5, bias=4),
            Node("e", "affine", ("d",), 4097, scale=2, bias=-3),
        ),
        outputs=("e",),
    )


def test_compiler():
    compiled = compile_graph(fixture(), [2, 8, 11], 3, slot_count=12)
    assert len(compiled.instructions) == 12
    assert len(compiled.instructions[0]) == 10
    assert compiled.offsets["x"] == compiled.offsets["b"]
    assert compiled.arena_elements == 3 * 4160
    assert compiled.instructions[8][0][0] == 3
    assert compiled.instructions[8][1][:2] == [4, 0]
    assert compiled.instructions[0][0][0] == 0
    assert compiled.describe()["waves"][2]["inputs"] == ["a", "b"]
    try:
        compile_graph(fixture(), [2, 8], 3)
    except ValueError:
        pass
    else:
        raise AssertionError("nonresident grid accepted")
    try:
        compile_graph(fixture(), [2, 8, 11], 3, max_arena_elements=100)
    except ValueError:
        pass
    else:
        raise AssertionError("arena budget ignored")
    bad = Graph({"x": 4}, (Node("a", "affine", ("b",), 4),), ("a",))
    try:
        compile_graph(bad, [2, 8, 11], 3)
    except ValueError:
        pass
    else:
        raise AssertionError("forward dependency accepted")
    try:
        compile_graph(fixture(), [2, 8, 11], 3,
                      operations={"affine": Operation(3, 1, 1, Resources(shared_bytes=1)),
                                  "add": Operation(3, 2, 2, Resources())})
    except ValueError:
        pass
    else:
        raise AssertionError("extra shared-memory request accepted")


def test_gpu():
    import torch
    from mk_h3 import run

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this test targets RTX PRO 6000 SM120")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    probe = torch.zeros((1024, 1, 32), device="cuda", dtype=torch.int32)
    probe[:, 0, 0] = 1
    probe_times = torch.zeros((1024, 1, 128), device="cuda", dtype=torch.int32)
    counts = torch.zeros(sm_count, device="cuda", dtype=torch.int32)
    ids = torch.full((sm_count,), -1, device="cuda", dtype=torch.int32)
    rendezvous = torch.zeros(1, device="cuda", dtype=torch.int32)
    dummy = torch.zeros(1, device="cuda", dtype=torch.float32)
    dummy_int = torch.zeros(1, device="cuda", dtype=torch.int32)
    run(probe, probe_times, counts, ids, rendezvous, dummy, dummy_int,
        dummy_int, sm_count, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    measured = ids.cpu().tolist()
    assert len(set(measured)) == sm_count and max(measured) < 1024
    compiled = compile_graph(fixture(), measured, sm_count, slot_count=1024)
    instructions, timings, arena, waves = compiled.tensors()
    error = torch.zeros(1, device="cuda", dtype=torch.int32)
    x = torch.arange(4097, device="cuda", dtype=torch.float32) / 32
    expected_a = x * 2 + 1
    expected_b = expected_a * 3 - 2
    expected = ((expected_a + expected_b) * 0.5 + 4) * 2 - 3
    for _ in range(2):
        arena.zero_()
        arena[compiled.offsets["x"]:compiled.offsets["x"] + 4097] = x
        waves.zero_()
        error.zero_()
        run(instructions, timings, counts, ids, rendezvous, arena, waves,
            error, sm_count, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        assert error.item() == 0, f"wave barrier timed out: {error.item()}"
        assert torch.equal(waves, torch.full_like(waves, sm_count)), waves.cpu()
        got = arena[compiled.offsets["e"]:compiled.offsets["e"] + 4097]
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    print(f"PASS: {sm_count} SM120 CTAs, 5 graph waves, dependencies, arena reuse, repeat launch")


if __name__ == "__main__":
    test_compiler()
    test_gpu()
