#include <cstdint>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include "h3.cuh"
#include "ops/smoke.cuh"
#include "ops/projection.cuh"
#include "ops/qknorm_rope.cuh"
#include "ops/graph.cuh"
#include "ops/sparse_attention.cuh"
#include "ops/vsa_pool.cuh"
#include "ops/vsa_scores.cuh"
#include "ops/vsa_topk.cuh"
#include "ops/vsa_value_pool.cuh"
#include "ops/vsa_compress.cuh"
#include "cutlass/util/packed_stride.hpp"
#include "pyutils/pyutils.cuh"

namespace py = pybind11;

namespace h3 {
// Keep the ordinary runtime ABI unchanged. This projection-only binding adds
// a device-resident, host-prepared CUTLASS Params record.
struct projection_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> rendezvous_count;
    kittens::gl<float, 1, 1, 1, -1> graph_arena;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    kittens::gl<int, 1, 1, 1, -1> projection_params;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct attention_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> rendezvous_count;
    kittens::gl<float, 1, 1, 1, -1> graph_arena;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    kittens::gl<int, 1, 1, 1, -1> projection_params;
    kittens::gl<int, 1, 1, 1, -1> q_input;
    kittens::gl<int, 1, 1, 1, -1> k_input;
    kittens::gl<int, 1, 1, 1, -1> q_output;
    kittens::gl<int, 1, 1, 1, -1> k_output;
    kittens::gl<int, 1, 1, 1, -1> q_weight;
    kittens::gl<int, 1, 1, 1, -1> k_weight;
    kittens::gl<int, 1, 1, 1, -1> rope_cos;
    kittens::gl<int, 1, 1, 1, -1> rope_sin;
    kittens::gl<int, 1, 1, 1, -1> v_input;
    kittens::gl<int, 1, 1, 1, -1> gate_input;
    kittens::gl<int, 1, 1, 1, -1> v_output;
    kittens::gl<int, 1, 1, 1, -1> gate_output;
    kittens::gl<int, 1, 1, 1, -1> tile_index;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct projection_sparse_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> rendezvous_count;
    kittens::gl<float, 1, 1, 1, -1> graph_arena;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    kittens::gl<int, 1, 1, 1, -1> projection_params;
    kittens::gl<int, 1, 1, 1, -1> q_input;
    kittens::gl<int, 1, 1, 1, -1> k_input;
    kittens::gl<int, 1, 1, 1, -1> q_output;
    kittens::gl<int, 1, 1, 1, -1> k_output;
    kittens::gl<int, 1, 1, 1, -1> q_weight;
    kittens::gl<int, 1, 1, 1, -1> k_weight;
    kittens::gl<int, 1, 1, 1, -1> rope_cos;
    kittens::gl<int, 1, 1, 1, -1> rope_sin;
    kittens::gl<int, 1, 1, 1, -1> v_input;
    kittens::gl<int, 1, 1, 1, -1> gate_input;
    kittens::gl<int, 1, 1, 1, -1> v_output;
    kittens::gl<int, 1, 1, 1, -1> gate_output;
    kittens::gl<int, 1, 1, 1, -1> tile_index;
    int sm_count;
    kittens::gl<int, 1, 1, 1, -1> sparse_mask;
    kittens::gl<int, 1, 1, 1, -1> valid_lengths;
    kittens::gl<int, 1, 1, 1, -1> sparse_output;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct sparse_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> rendezvous_count;
    kittens::gl<float, 1, 1, 1, -1> graph_arena;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    kittens::gl<int, 1, 1, 1, -1> q_output;
    kittens::gl<int, 1, 1, 1, -1> k_output;
    kittens::gl<int, 1, 1, 1, -1> v_output;
    kittens::gl<int, 1, 1, 1, -1> sparse_mask;
    kittens::gl<int, 1, 1, 1, -1> valid_lengths;
    kittens::gl<int, 1, 1, 1, -1> sparse_output;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct vsa_pool_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> rendezvous_count;
    kittens::gl<float, 1, 1, 1, -1> graph_arena;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    kittens::gl<int, 1, 1, 1, -1> q_output;
    kittens::gl<int, 1, 1, 1, -1> k_output;
    kittens::gl<int, 1, 1, 1, -1> valid_lengths;
    kittens::gl<int, 1, 1, 1, -1> q_pool;
    kittens::gl<int, 1, 1, 1, -1> k_pool;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct vsa_scores_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> rendezvous_count;
    kittens::gl<float, 1, 1, 1, -1> graph_arena;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    kittens::gl<int, 1, 1, 1, -1> q_pool;
    kittens::gl<int, 1, 1, 1, -1> k_pool;
    kittens::gl<int, 1, 1, 1, -1> scores;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct vsa_topk_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> rendezvous_count;
    kittens::gl<float, 1, 1, 1, -1> graph_arena;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    kittens::gl<int, 1, 1, 1, -1> scores;
    kittens::gl<int, 1, 1, 1, -1> sparse_mask;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct vsa_dynamic_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> rendezvous_count;
    kittens::gl<float, 1, 1, 1, -1> graph_arena;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    kittens::gl<int, 1, 1, 1, -1> q_output;
    kittens::gl<int, 1, 1, 1, -1> k_output;
    kittens::gl<int, 1, 1, 1, -1> v_output;
    kittens::gl<int, 1, 1, 1, -1> valid_lengths;
    kittens::gl<int, 1, 1, 1, -1> q_pool;
    kittens::gl<int, 1, 1, 1, -1> k_pool;
    kittens::gl<int, 1, 1, 1, -1> v_pool;
    kittens::gl<int, 1, 1, 1, -1> scores;
    kittens::gl<int, 1, 1, 1, -1> sparse_mask;
    kittens::gl<int, 1, 1, 1, -1> sparse_output;
    kittens::gl<int, 1, 1, 1, -1> gate_output;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};
} // namespace h3

static h3::projection::Kernel::Arguments arguments(
    std::uintptr_t a, std::uintptr_t b, std::uintptr_t sf_a,
    std::uintptr_t sf_b, std::uintptr_t d, int m, int n, int k,
    float alpha, int sm_count) {
    using Kernel = h3::projection::Kernel;
    using Mainloop = h3::projection::Mainloop;
    using A = h3::projection::A;
    using B = h3::projection::B;
    using D = h3::projection::D;
    if (m <= 0 || n <= 0 || k <= 0 || n % 128 || k % 128 || sm_count <= 0)
        throw std::invalid_argument("invalid SM120 projection shape or SM count");
    auto stride_a = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1});
    auto stride_b = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1});
    auto stride_d = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1});
    using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
    auto shape = cute::make_shape(m, n, k, 1);
    auto layout_sf_a = ScaleConfig::tile_atom_to_shape_SFA(shape);
    auto layout_sf_b = ScaleConfig::tile_atom_to_shape_SFB(shape);
    auto args = typename Kernel::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {reinterpret_cast<A::DataType*>(a), stride_a,
         reinterpret_cast<B::DataType*>(b), stride_b,
         reinterpret_cast<A::ScaleFactorType*>(sf_a), layout_sf_a,
         reinterpret_cast<B::ScaleFactorType*>(sf_b), layout_sf_b},
        {{alpha, 0.0f}, reinterpret_cast<D*>(d), stride_d,
         reinterpret_cast<D*>(d), stride_d}};
    args.hw_info.sm_count = sm_count;
    return args;
}

PYBIND11_MODULE(mk_h3_projection, m) {
    using Kernel = h3::projection::Kernel;
    m.def("params_words", []() { return (sizeof(Kernel::Params) + 3) / 4; });
    m.def("resources", [](int rows, int cols, int depth) {
        py::dict out;
        out["threads_per_cta"] = Kernel::MaxThreadsPerBlock;
        out["shared_bytes"] = sizeof(Kernel::SharedStorage);
        out["params_bytes"] = sizeof(Kernel::Params);
        out["tile_count"] = ((rows + 127) / 128) * ((cols + 127) / 128);
        return out;
    });
    m.def("prepare", [](std::uintptr_t params_device, std::uintptr_t a,
                         std::uintptr_t b, std::uintptr_t sf_a,
                         std::uintptr_t sf_b, std::uintptr_t d,
                         int rows, int cols, int depth, float alpha,
                         int sm_count) {
        if (!params_device || !a || !b || !sf_a || !sf_b || !d)
            throw std::invalid_argument("null projection pointer");
        auto args = arguments(a, b, sf_a, sf_b, d, rows, cols, depth,
                              alpha, sm_count);
        if (!Kernel::can_implement(args))
            throw std::runtime_error("CUTLASS rejected Hazy projection arguments");
        auto params = Kernel::to_underlying_arguments(args, nullptr);
        auto error = cudaMemcpy(reinterpret_cast<void*>(params_device),
                                &params, sizeof(params), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    });
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::projection_globals,
                       h3::smoke_op, h3::projection_op>>(
        m, "run", &h3::projection_globals::instructions,
        &h3::projection_globals::timings,
        &h3::projection_globals::completion_counts,
        &h3::projection_globals::physical_sm_ids,
        &h3::projection_globals::rendezvous_count,
        &h3::projection_globals::graph_arena,
        &h3::projection_globals::wave_counts,
        &h3::projection_globals::graph_error,
        &h3::projection_globals::projection_params,
        &h3::projection_globals::sm_count);
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::attention_globals,
                       h3::smoke_op, h3::projection_op, h3::qknorm_rope_op,
                       h3::wave_barrier_op>>(
        m, "run_attention", &h3::attention_globals::instructions,
        &h3::attention_globals::timings,
        &h3::attention_globals::completion_counts,
        &h3::attention_globals::physical_sm_ids,
        &h3::attention_globals::rendezvous_count,
        &h3::attention_globals::graph_arena,
        &h3::attention_globals::wave_counts,
        &h3::attention_globals::graph_error,
        &h3::attention_globals::projection_params,
        &h3::attention_globals::q_input,
        &h3::attention_globals::k_input,
        &h3::attention_globals::q_output,
        &h3::attention_globals::k_output,
        &h3::attention_globals::q_weight,
        &h3::attention_globals::k_weight,
        &h3::attention_globals::rope_cos,
        &h3::attention_globals::rope_sin,
        &h3::attention_globals::v_input,
        &h3::attention_globals::gate_input,
        &h3::attention_globals::v_output,
        &h3::attention_globals::gate_output,
        &h3::attention_globals::tile_index,
        &h3::attention_globals::sm_count);
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::sparse_globals,
                       h3::smoke_op, h3::sparse_attention_op>>(
        m, "run_sparse", &h3::sparse_globals::instructions,
        &h3::sparse_globals::timings,
        &h3::sparse_globals::completion_counts,
        &h3::sparse_globals::physical_sm_ids,
        &h3::sparse_globals::rendezvous_count,
        &h3::sparse_globals::graph_arena,
        &h3::sparse_globals::wave_counts,
        &h3::sparse_globals::graph_error,
        &h3::sparse_globals::q_output,
        &h3::sparse_globals::k_output,
        &h3::sparse_globals::v_output,
        &h3::sparse_globals::sparse_mask,
        &h3::sparse_globals::valid_lengths,
        &h3::sparse_globals::sparse_output,
        &h3::sparse_globals::sm_count);
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::vsa_pool_globals,
                       h3::smoke_op, h3::vsa_pool_op>>(
        m, "run_vsa_pool", &h3::vsa_pool_globals::instructions,
        &h3::vsa_pool_globals::timings,
        &h3::vsa_pool_globals::completion_counts,
        &h3::vsa_pool_globals::physical_sm_ids,
        &h3::vsa_pool_globals::rendezvous_count,
        &h3::vsa_pool_globals::graph_arena,
        &h3::vsa_pool_globals::wave_counts,
        &h3::vsa_pool_globals::graph_error,
        &h3::vsa_pool_globals::q_output,
        &h3::vsa_pool_globals::k_output,
        &h3::vsa_pool_globals::valid_lengths,
        &h3::vsa_pool_globals::q_pool,
        &h3::vsa_pool_globals::k_pool,
        &h3::vsa_pool_globals::sm_count);
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::vsa_scores_globals,
                       h3::smoke_op, h3::vsa_scores_op>>(
        m, "run_vsa_scores", &h3::vsa_scores_globals::instructions,
        &h3::vsa_scores_globals::timings,
        &h3::vsa_scores_globals::completion_counts,
        &h3::vsa_scores_globals::physical_sm_ids,
        &h3::vsa_scores_globals::rendezvous_count,
        &h3::vsa_scores_globals::graph_arena,
        &h3::vsa_scores_globals::wave_counts,
        &h3::vsa_scores_globals::graph_error,
        &h3::vsa_scores_globals::q_pool,
        &h3::vsa_scores_globals::k_pool,
        &h3::vsa_scores_globals::scores,
        &h3::vsa_scores_globals::sm_count);
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::vsa_topk_globals,
                       h3::smoke_op, h3::vsa_topk_op>>(
        m, "run_vsa_topk", &h3::vsa_topk_globals::instructions,
        &h3::vsa_topk_globals::timings,
        &h3::vsa_topk_globals::completion_counts,
        &h3::vsa_topk_globals::physical_sm_ids,
        &h3::vsa_topk_globals::rendezvous_count,
        &h3::vsa_topk_globals::graph_arena,
        &h3::vsa_topk_globals::wave_counts,
        &h3::vsa_topk_globals::graph_error,
        &h3::vsa_topk_globals::scores,
        &h3::vsa_topk_globals::sparse_mask,
        &h3::vsa_topk_globals::sm_count);
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::vsa_dynamic_globals,
                       h3::smoke_op, h3::vsa_pool_op, h3::vsa_scores_op,
                       h3::vsa_topk_op, h3::wave_barrier_op,
                       h3::sparse_attention_op>>(
        m, "run_vsa_dynamic", &h3::vsa_dynamic_globals::instructions,
        &h3::vsa_dynamic_globals::timings,
        &h3::vsa_dynamic_globals::completion_counts,
        &h3::vsa_dynamic_globals::physical_sm_ids,
        &h3::vsa_dynamic_globals::rendezvous_count,
        &h3::vsa_dynamic_globals::graph_arena,
        &h3::vsa_dynamic_globals::wave_counts,
        &h3::vsa_dynamic_globals::graph_error,
        &h3::vsa_dynamic_globals::q_output,
        &h3::vsa_dynamic_globals::k_output,
        &h3::vsa_dynamic_globals::v_output,
        &h3::vsa_dynamic_globals::valid_lengths,
        &h3::vsa_dynamic_globals::q_pool,
        &h3::vsa_dynamic_globals::k_pool,
        &h3::vsa_dynamic_globals::v_pool,
        &h3::vsa_dynamic_globals::scores,
        &h3::vsa_dynamic_globals::sparse_mask,
        &h3::vsa_dynamic_globals::sparse_output,
        &h3::vsa_dynamic_globals::gate_output,
        &h3::vsa_dynamic_globals::sm_count);
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::vsa_dynamic_globals,
                       h3::smoke_op, h3::vsa_pool_op, h3::vsa_value_pool_op,
                       h3::vsa_scores_op, h3::vsa_topk_op,
                       h3::wave_barrier_op, h3::sparse_attention_op,
                       h3::vsa_compress_op>>(
        m, "run_vsa_full", &h3::vsa_dynamic_globals::instructions,
        &h3::vsa_dynamic_globals::timings,
        &h3::vsa_dynamic_globals::completion_counts,
        &h3::vsa_dynamic_globals::physical_sm_ids,
        &h3::vsa_dynamic_globals::rendezvous_count,
        &h3::vsa_dynamic_globals::graph_arena,
        &h3::vsa_dynamic_globals::wave_counts,
        &h3::vsa_dynamic_globals::graph_error,
        &h3::vsa_dynamic_globals::q_output,
        &h3::vsa_dynamic_globals::k_output,
        &h3::vsa_dynamic_globals::v_output,
        &h3::vsa_dynamic_globals::valid_lengths,
        &h3::vsa_dynamic_globals::q_pool,
        &h3::vsa_dynamic_globals::k_pool,
        &h3::vsa_dynamic_globals::v_pool,
        &h3::vsa_dynamic_globals::scores,
        &h3::vsa_dynamic_globals::sparse_mask,
        &h3::vsa_dynamic_globals::sparse_output,
        &h3::vsa_dynamic_globals::gate_output,
        &h3::vsa_dynamic_globals::sm_count);
    kittens::py::bind_kernel<
        megakernel::mk<h3::config, h3::projection_sparse_globals,
                       h3::smoke_op, h3::projection_op, h3::qknorm_rope_op,
                       h3::wave_barrier_op, h3::sparse_attention_op>>(
        m, "run_projection_sparse", &h3::projection_sparse_globals::instructions,
        &h3::projection_sparse_globals::timings,
        &h3::projection_sparse_globals::completion_counts,
        &h3::projection_sparse_globals::physical_sm_ids,
        &h3::projection_sparse_globals::rendezvous_count,
        &h3::projection_sparse_globals::graph_arena,
        &h3::projection_sparse_globals::wave_counts,
        &h3::projection_sparse_globals::graph_error,
        &h3::projection_sparse_globals::projection_params,
        &h3::projection_sparse_globals::q_input,
        &h3::projection_sparse_globals::k_input,
        &h3::projection_sparse_globals::q_output,
        &h3::projection_sparse_globals::k_output,
        &h3::projection_sparse_globals::q_weight,
        &h3::projection_sparse_globals::k_weight,
        &h3::projection_sparse_globals::rope_cos,
        &h3::projection_sparse_globals::rope_sin,
        &h3::projection_sparse_globals::v_input,
        &h3::projection_sparse_globals::gate_input,
        &h3::projection_sparse_globals::v_output,
        &h3::projection_sparse_globals::gate_output,
        &h3::projection_sparse_globals::tile_index,
        &h3::projection_sparse_globals::sm_count,
        &h3::projection_sparse_globals::sparse_mask,
        &h3::projection_sparse_globals::valid_lengths,
        &h3::projection_sparse_globals::sparse_output);
}
