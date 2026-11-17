#pragma once

#ifndef H3_SM120
#error "The CUTLASS projection instruction requires SM120"
#endif

#include "../h3.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

// CUTLASS 4.4.2 provides this scheduler for SM100 but omits the SM120 tag.
// The scheduler uses ordinary block IDs and a grid-size stride; it does not
// issue SM100 instructions. Hazy needs this form for its 188-CTA resident grid.
namespace cutlass::gemm::kernel::detail {
template <class TileShape, class ClusterShape, uint32_t Stages>
struct TileSchedulerSelector<cutlass::gemm::StaticPersistentScheduler,
                             cutlass::arch::Sm120, TileShape, ClusterShape,
                             Stages> {
    using Scheduler = StaticPersistentTileScheduler100;
};
} // namespace cutlass::gemm::kernel::detail

namespace h3::projection {
using namespace cute;
using A = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using B = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using D = cutlass::bfloat16_t;
using Arch = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, D, cutlass::layout::RowMajor, 8,
    D, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, OpClass, A, cutlass::layout::RowMajor, 32,
    B, cutlass::layout::ColumnMajor, 32, float, Tile, Cluster,
    cutlass::gemm::collective::StageCount<2>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue,
    cutlass::gemm::StaticPersistentScheduler>;

static_assert(Kernel::MaxThreadsPerBlock == config::NUM_THREADS);
static_assert(sizeof(Kernel::SharedStorage) <=
              config::DYNAMIC_SHARED_MEMORY - 1024);

template <typename G>
__device__ inline void execute(const G &g, state &s) {
    // Every worker has finished the previous instruction; the loader has
    // returned all five pages. CUTLASS owns the aligned dynamic region until
    // the second barrier. Its own role split uses all twelve Hazy warps.
    __syncthreads();
    auto *params = reinterpret_cast<const Kernel::Params *>(
        g.projection_params.raw_ptr);
    Kernel{}(params[s.instruction()[1]], reinterpret_cast<char *>(&s.pages));
    __syncthreads();
}
} // namespace h3::projection

namespace h3 {
struct projection_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::projection_gemm;
    struct controller : megakernel::NoOp<config>::controller {
        template <typename G>
        static __device__ void run(const G &g, state &s) {
            projection::execute(g, s);
        }
    };
    struct loader {
        template <typename G>
        static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::loader::run(g, s);
            projection::execute(g, s);
        }
    };
    struct launcher {
        template <typename G>
        static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::launcher::run(g, s);
            projection::execute(g, s);
        }
    };
    struct consumer {
        template <typename G>
        static __device__ void run(const G &g, state &s) {
            projection::execute(g, s);
        }
    };
    struct storer {
        template <typename G>
        static __device__ void run(const G &g, state &s) {
            projection::execute(g, s);
        }
    };
};
} // namespace h3
