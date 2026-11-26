// Python binding for the H3 megakernel: one kernel that dispatches every op by opcode.
// Add each new op to the mk<...> list below.

#include "h3.cuh"
#include "ops/add_one.cuh"
#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // argument order must match the member order in globals_t
    // template arg is the kernel to launch (megakernel)
    // m: pybind module
    // name: python func name
    // remaining python args, which are the input weights and output weight storage
    kittens::py::bind_kernel<megakernel::mk<config, globals_t,
                                      AddOneOp>>(
        m, "h3",
        &globals_t::instructions,
        &globals_t::timings,
        &globals_t::input_weights,
        &globals_t::output_weights);
}
