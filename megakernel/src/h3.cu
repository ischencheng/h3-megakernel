#include "h3.cuh"
#include "ops/smoke.cuh"
#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(mk_h3, m) {
    m.doc() = "H3 megakernel runtime scaffold; model operations not implemented";
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::globals, h3::smoke_op>>(
        m, "run", &h3::globals::instructions, &h3::globals::timings,
        &h3::globals::completion_counts, &h3::globals::sm_count);
}
