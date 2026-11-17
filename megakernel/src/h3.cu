#include "h3.cuh"
#include "ops/smoke.cuh"
#include "ops/rendezvous.cuh"
#include "ops/graph.cuh"
#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(mk_h3, m) {
    m.doc() = "H3 megakernel runtime scaffold; model operations not implemented";
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::globals, h3::smoke_op, h3::rendezvous_op,
                       h3::graph_task_op, h3::wave_barrier_op>>(
        m, "run", &h3::globals::instructions, &h3::globals::timings,
        &h3::globals::completion_counts, &h3::globals::physical_sm_ids,
        &h3::globals::rendezvous_count, &h3::globals::graph_arena,
        &h3::globals::wave_counts, &h3::globals::graph_error,
        &h3::globals::sm_count);
}
