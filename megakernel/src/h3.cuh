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

// constants for h3 architecture

#define RTX6000_SM_COUNT 188

#define HIDDEN_DIM 5376
#define H3_NUM_BLOCKS 50
#define H3_NUM_STEPS 8
#define H3_HIDDEN_DIM 5376
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
    // types 
    using instruction_layout = megakernel::instruction_layout<config>;
    using test_t = kittens::gl<kittens::bf16, 1, -1, -1, H3_HIDDEN_DIM, kittens::st_bf<128, 64>>;

    test_t input_weights;
    test_t output_weights;

}