// Standalone SM120 feasibility probe, adapted from NVIDIA CUTLASS example 79a.
// It intentionally does not run inside Hazy's persistent kernel yet.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace py = pybind11;
using namespace cute;

using A = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using B = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using D = cutlass::bfloat16_t;
using Arch = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, OpClass, ThreadBlockShape, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, D, cutlass::layout::RowMajor, 8,
    D, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, OpClass, A, cutlass::layout::RowMajor, 32,
    B, cutlass::layout::ColumnMajor, 32, float, ThreadBlockShape, Cluster,
    // Two pipeline stages leave shared memory for Hazy's control state when
    // this collective is eventually embedded in the persistent CTA.
    cutlass::gemm::collective::StageCount<2>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

static Gemm::Arguments arguments(
    std::uintptr_t a, std::uintptr_t b, std::uintptr_t sf_a,
    std::uintptr_t sf_b, std::uintptr_t d, int m, int n, int k, float alpha) {
  if (m <= 0 || n <= 0 || k <= 0 || n % 128 != 0 || k % 128 != 0)
    throw std::invalid_argument("M must be positive; N and K must be positive multiples of 128");
  auto stride_a = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1});
  auto stride_d = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1});
  using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
  auto shape = cute::make_shape(m, n, k, 1);
  auto layout_sf_a = ScaleConfig::tile_atom_to_shape_SFA(shape);
  auto layout_sf_b = ScaleConfig::tile_atom_to_shape_SFB(shape);
  return {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<A::DataType*>(a), stride_a,
       reinterpret_cast<B::DataType*>(b), stride_b,
       reinterpret_cast<A::ScaleFactorType*>(sf_a), layout_sf_a,
       reinterpret_cast<B::ScaleFactorType*>(sf_b), layout_sf_b},
      {{alpha, 0.0f}, reinterpret_cast<D*>(d), stride_d,
       reinterpret_cast<D*>(d), stride_d}};
}

static py::dict resources(int m, int n, int k) {
  auto args = arguments(0, 0, 0, 0, 0, m, n, k, 1.0f);
  py::dict out;
  out["threads_per_cta"] = Kernel::MaxThreadsPerBlock;
  out["shared_bytes"] = sizeof(typename Kernel::SharedStorage);
  out["workspace_bytes"] = Gemm::get_workspace_size(args);
  out["grid_x"] = Gemm::get_grid_shape(args).x;
  out["grid_y"] = Gemm::get_grid_shape(args).y;
  out["can_implement"] = Gemm::can_implement(args) == cutlass::Status::kSuccess;
  return out;
}

static void run(std::uintptr_t a, std::uintptr_t b,
                std::uintptr_t sf_a, std::uintptr_t sf_b,
                std::uintptr_t d, std::uintptr_t workspace,
                int m, int n, int k, float alpha, std::uintptr_t stream) {
  auto args = arguments(a, b, sf_a, sf_b, d, m, n, k, alpha);
  if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
    throw std::runtime_error("CUTLASS rejected the NVFP4 projection arguments");
  if (Gemm::get_workspace_size(args) && !workspace)
    throw std::runtime_error("CUTLASS requires a preallocated workspace");
  Gemm gemm;
  auto status = gemm(args, reinterpret_cast<void*>(workspace),
                     reinterpret_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error("CUTLASS launch failed, status=" +
                             std::to_string(static_cast<int>(status)));
}

PYBIND11_MODULE(cutlass_projection, m) {
  m.doc() = "SM120 NVFP4 projection and CUTLASS resource probe";
  m.def("resources", &resources);
  m.def("run", &run);
}
