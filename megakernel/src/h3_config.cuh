#pragma once
#include "config.cuh"

namespace h3 {
// Reference H100/B200 configuration, NOT yet an RTX PRO 6000 SM120 port.
struct config : megakernel::default_config {
    static constexpr int GMEM_SPIN_LOOP_SLEEP_NANOS = 20;
#ifdef H3_SM120
    // SM120 permits at most 99 KiB/block. Leave margin for the aligned
    // dynamic page base and keep the upstream two-slot instruction protocol.
    static constexpr int MAX_SHARED_MEMORY = 100000;
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 5, "SM120 page configuration changed");
    static_assert(STATIC_SHARED_MEMORY + DYNAMIC_SHARED_MEMORY <= 99 * 1024);
#endif
};
} // namespace h3
