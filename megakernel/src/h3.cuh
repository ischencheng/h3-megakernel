#pragma once

// third party recognition: parts inspired by demo from Hazy Research
// See megakernel/include for LICENSE


#include "kittens.cuh" // for thunderkittens
#include "megakernel.cuh" // for our megakernal runtime
#include <iostream>

// operations
#define OPCODE_ADDONE 1 // this is for practice

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


using config =

// use template if necessary?
struct h3_globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;

    kittens::gl<float, 1, 1, 1, -1> weights;

    int sm_count;

    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() {return config::DYNAMIC_SHARED_MEMORY; }
}