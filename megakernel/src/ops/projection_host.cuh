#pragma once

// Host-side preparation of CUTLASS SM120 NVFP4 projection parameters, shared
// by the projection gates (h3_projection.cu) and the full DiT (h3_dit.cu).
// The device-resident Params records are written before launch; the kernel
// performs no allocation or launch of its own.

#include <cstdint>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include "projection.cuh"
#include "cutlass/util/packed_stride.hpp"

namespace h3::projection {
inline Kernel::Arguments arguments(
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

inline void bind_host(pybind11::module_ &m) {
    namespace py = pybind11;
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
}
} // namespace h3::projection
