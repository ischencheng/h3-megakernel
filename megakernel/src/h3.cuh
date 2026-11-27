#pragma once

// third party recognition: parts inspired by demo from Hazy Research
// See megakernel/include for LICENSE


#include "kittens.cuh" // for thunderkittens
#include "megakernel.cuh" // for our megakernal runtime
#include <iostream>

using config = megakernel::default_config;

// operations
#define OPCODE_TEST 1
#define OPCODE_ADD_ONE 2 // for writing with add_one
#define OPCODE_NORM_MOD_QUANT 3

// constants for h3 architecture

#define RTX6000_SM_COUNT 188

#define HIDDEN_DIM 5376 // pref HIDDEN_DIM
#define H3_HIDDEN_DIM 5376 // repeat but leaving in in case i use both in different places
#define CHUNK_COLS 64 // how many columns are in one chunk (16 rows, 64 of hidden dim columns)
#define NUM_CHUNKS (HIDDEN_DIM / CHUNK_COLS)
#define NORM_EPS 1e-5f // the epsilon that fastvideo uses for RMSNorm calcs


#define H3_NUM_BLOCKS 50
#define H3_NUM_STEPS 8
#define H3_FFN_DIM 14336
#define H3_HEAD_DIM 128
#define H3_NUM_ATTENTION_HEADS 56
#define H3_NUM_MODALITIES 3
#define H3_TEXT_EMBED_DIM 5120
#define H3_VIDEO_LATENT_CHANNELS 96
#define H3_AUDIO_LATENT_CHANNELS 32

#define H3_VIDEO_TOKENS 37296
#define H3_AUDIO_TOKENS 414
#define H3_NUM_TILES 668


struct globals_t {
    using instruction_layout = megakernel::instruction_layout<config>;
    using timing_layout = megakernel::timing_layout<config>;

    // kittens::gl encodes device pointer + 4D shape, Each dim either fixed at compile or -1, meaning it
    // needs to be passed in construction
    using test_t = kittens::gl<kittens::bf16, 1, 1, -1, H3_HIDDEN_DIM, kittens::st_bf<128, 64>>; // [rows, 5376]

    using hidden_t = kittens::gl<kittens::bf16, 1, 1, -1, H3_HIDDEN_DIM, kittens::st_bf<128, 64>>; // [S, 5376], hidden dim for DiT


    // usage: g.instructions[{sm_id, i, 0}] and g.timings[{sm_id, i, 0}]
    instruction_layout instructions; // [num_sms, num_instructions, INSTRUCTION_WIDTH]
    timing_layout timings;           // [num_sms, num_instructions, TIMING_WIDTH]

    test_t input_weights;
    test_t output_weights;

    hidden_t x; // hidden weights of Dit

    // launch shape for kittens::py::bind_kernel: one block per SM, since the
    // runtime indexes instructions by get_smid(), not blockIdx.x
    dim3 grid() { return dim3(instructions.depth()); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};