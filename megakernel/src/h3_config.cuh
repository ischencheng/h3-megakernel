#pragma once
#include "config.cuh"

namespace h3 {
// Reference H100/B200 configuration, NOT yet an RTX PRO 6000 SM120 port.
struct config : megakernel::default_config {
    static constexpr int GMEM_SPIN_LOOP_SLEEP_NANOS = 20;
};
} // namespace h3
