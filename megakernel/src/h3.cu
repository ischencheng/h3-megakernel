// Python binding for the H3 megakernel: one kernel that dispatches every op by opcode.
// Add each new op to the mk<...> list below.

#include "h3.cuh"
#include "ops/add_one.cuh"
#include "ops/norm_mod_quant.cuh"
#include "pyutils/pyutils.cuh"
#include <cute/arch/mma_sm120.hpp> // CUTLASS's NVFP4 mma

// fail the build if the FP4 mma not available
#if defined(__CUDA_ARCH__) && !defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
#error "mxf4nvf4 block-scaled mma.sync not enabled: compile for sm_120a"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // argument order must match the member order in globals_t
    // template arg is the kernel to launch (megakernel)
    // m: pybind module
    // name: python func name
    // remaining python args, which are the input weights and output weight storage
    kittens::py::bind_kernel<megakernel::mk<config, globals_t,
                                      AddOneOp, NormModQuantOp>>(
        m, "h3",
        &globals_t::instructions, // these are the 4 args
        &globals_t::timings,
        &globals_t::input_weights,
        &globals_t::output_weights,
        &globals_t::x,
        &globals_t::norm_weight_vec,
        &globals_t::adaln_table,
        &globals_t::adaln_index,
        &globals_t::packed,
        &globals_t::scales);
}
