/*
First operation in the DiT block: covers the adaLN norm and fp4 quantization
*/

#pragma once
#include "../h3.cuh"

struct NormModQuantOp : megakernel::NoOp<config> {
    static constexpr int opcode = OPCODE_NORM_MOD_QUANT;
    
};