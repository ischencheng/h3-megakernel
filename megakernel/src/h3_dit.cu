// Full FastH3 DiT instruction set in one Hazy launch (SM120).
//
// `run` executes a host-compiled per-SM instruction stream over a device
// pointer table. Shapes, weights, schedules and buffer lifetimes are decided
// by megakernel/h3_dit.py; nothing checkpoint specific is compiled in here
// beyond the 128-channel head geometry of the attention operations.

#include "h3.cuh"
#include "ops/smoke.cuh"
#include "ops/projection_host.cuh"
#include "ops/dit_ops.cuh"
#include "ops/dit_attention.cuh"
#include "pyutils/pyutils.cuh"

namespace h3 {
struct dit_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    kittens::gl<int, 1, 1, 1, -1> projection_params;
    // int64 device addresses, one per buffer index used by the instructions.
    kittens::gl<int, 1, 1, 1, -1> buffers;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

namespace dit {
// PROJECTION: 1 parameter record index. The CUTLASS collective owns the
// aligned dynamic page region between the two CTA barriers of the wrapper.
struct projection_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        auto *params = reinterpret_cast<const projection::Kernel::Params *>(
            g.projection_params.raw_ptr);
        projection::Kernel{}(params[s.instruction()[1]],
                             reinterpret_cast<char *>(&s.pages));
    }
};
} // namespace dit

using dit_projection_op = dit::all_warp_op<dit_projection, dit::projection_body>;
using dit_rmsnorm_mod_op = dit::all_warp_op<dit_rmsnorm_mod, dit::rmsnorm_mod_body>;
using dit_nvfp4_quant_op = dit::all_warp_op<dit_nvfp4_quant, dit::nvfp4_quant_body>;
using dit_dense_linear_op = dit::all_warp_op<dit_dense_linear, dit::dense_linear_body>;
using dit_elementwise_op = dit::all_warp_op<dit_elementwise, dit::elementwise_body>;
using dit_copy_rows_op = dit::all_warp_op<dit_copy_rows, dit::copy_rows_body>;
using dit_qknorm_tile_op = dit::all_warp_op<dit_qknorm_tile, dit::qknorm_tile_body>;
using dit_vsa_pool_op = dit::all_warp_op<dit_vsa_pool, dit::vsa_pool_body>;
using dit_vsa_scores_op = dit::all_warp_op<dit_vsa_scores, dit::vsa_scores_body>;
using dit_vsa_topk_op = dit::all_warp_op<dit_vsa_topk, dit::vsa_topk_body>;
using dit_sparse_attention_op =
    dit::all_warp_op<dit_sparse_attention, dit::sparse_attention_body>;
using dit_sparse_attention_mma_op =
    dit::all_warp_op<dit_sparse_attention_mma, dit::sparse_attention_mma_body>;
using dit_vsa_compress_op = dit::all_warp_op<dit_vsa_compress, dit::vsa_compress_body>;
} // namespace h3

PYBIND11_MODULE(mk_h3_dit, m) {
    m.doc() = "FastH3 DiT on the Hazy megakernel runtime (SM120)";
    h3::projection::bind_host(m);
    m.def("opcodes", []() {
        pybind11::dict out;
        out["SMOKE"] = static_cast<int>(h3::smoke);
        out["PROJECTION"] = static_cast<int>(h3::dit_projection);
        out["RMSNORM_MOD"] = static_cast<int>(h3::dit_rmsnorm_mod);
        out["NVFP4_QUANT"] = static_cast<int>(h3::dit_nvfp4_quant);
        out["DENSE_LINEAR"] = static_cast<int>(h3::dit_dense_linear);
        out["ELEMENTWISE"] = static_cast<int>(h3::dit_elementwise);
        out["COPY_ROWS"] = static_cast<int>(h3::dit_copy_rows);
        out["QKNORM_TILE"] = static_cast<int>(h3::dit_qknorm_tile);
        out["VSA_POOL"] = static_cast<int>(h3::dit_vsa_pool);
        out["VSA_SCORES"] = static_cast<int>(h3::dit_vsa_scores);
        out["VSA_TOPK"] = static_cast<int>(h3::dit_vsa_topk);
        out["SPARSE_ATTENTION"] = static_cast<int>(h3::dit_sparse_attention);
        out["SPARSE_ATTENTION_MMA"] = static_cast<int>(h3::dit_sparse_attention_mma);
        out["VSA_COMPRESS"] = static_cast<int>(h3::dit_vsa_compress);
        return out;
    });
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::dit_globals, h3::smoke_op,
                       h3::dit_projection_op, h3::dit_rmsnorm_mod_op,
                       h3::dit_nvfp4_quant_op, h3::dit_dense_linear_op,
                       h3::dit_elementwise_op, h3::dit_copy_rows_op,
                       h3::dit_qknorm_tile_op, h3::dit_vsa_pool_op,
                       h3::dit_vsa_scores_op, h3::dit_vsa_topk_op,
                       h3::dit_sparse_attention_op, h3::dit_sparse_attention_mma_op,
                       h3::dit_vsa_compress_op>>(
        m, "run", &h3::dit_globals::instructions, &h3::dit_globals::timings,
        &h3::dit_globals::completion_counts, &h3::dit_globals::physical_sm_ids,
        &h3::dit_globals::wave_counts, &h3::dit_globals::graph_error,
        &h3::dit_globals::projection_params, &h3::dit_globals::buffers,
        &h3::dit_globals::sm_count);
}
