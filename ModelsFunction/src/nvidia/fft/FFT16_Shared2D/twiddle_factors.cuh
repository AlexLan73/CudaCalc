/**
 * @file twiddle_factors.cuh
 * @brief Pre-computed twiddle factors for FFT16
 * 
 * OPTIMIZATION: Pre-computed sin/cos tables for maximum speed!
 * Eliminates runtime trigonometric calculations.
 */

#pragma once

namespace CudaCalc {

/**
 * @brief Pre-computed twiddle factors for FFT16
 * 
 * Twiddle factor W_N^k = exp(-i * 2π * k / N) = cos(angle) - i*sin(angle)
 * where angle = -2π * k / N
 * 
 * Storage: [stage][k] where stage = 0..3 for FFT16
 * - Stage 0: N=2,  k=0..0  (W_2^0)
 * - Stage 1: N=4,  k=0..1  (W_4^0, W_4^1)
 * - Stage 2: N=8,  k=0..3  (W_8^0, W_8^1, W_8^2, W_8^3)
 * - Stage 3: N=16, k=0..7  (W_16^0, W_16^1, ..., W_16^7)
 */

// Stage 0: W_2^k (k=0)
__constant__ float TWIDDLE_COS_STAGE0[1] = {1.0f};
__constant__ float TWIDDLE_SIN_STAGE0[1] = {0.0f};

// Stage 1: W_4^k (k=0,1)
__constant__ float TWIDDLE_COS_STAGE1[2] = {
    1.0f,                    // W_4^0 = 1
    0.0f                     // W_4^1 = -i → cos = 0
};
__constant__ float TWIDDLE_SIN_STAGE1[2] = {
    0.0f,                    // W_4^0 = 1
    -1.0f                    // W_4^1 = -i → sin = -1
};

// Stage 2: W_8^k (k=0,1,2,3)
__constant__ float TWIDDLE_COS_STAGE2[4] = {
    1.0f,                    // W_8^0
    0.70710678118f,          // W_8^1 = cos(-π/4) = √2/2
    0.0f,                    // W_8^2 = -i
    -0.70710678118f          // W_8^3 = cos(-3π/4)
};
__constant__ float TWIDDLE_SIN_STAGE2[4] = {
    0.0f,                    // W_8^0
    -0.70710678118f,         // W_8^1 = sin(-π/4) = -√2/2
    -1.0f,                   // W_8^2 = -i
    -0.70710678118f          // W_8^3 = sin(-3π/4)
};

// Stage 3: W_16^k (k=0,1,2,3,4,5,6,7)
__constant__ float TWIDDLE_COS_STAGE3[8] = {
    1.0f,                    // W_16^0 = cos(0)
    0.92387953251f,          // W_16^1 = cos(-π/8)
    0.70710678118f,          // W_16^2 = cos(-π/4) = √2/2
    0.38268343236f,          // W_16^3 = cos(-3π/8)
    0.0f,                    // W_16^4 = cos(-π/2) = -i
    -0.38268343236f,         // W_16^5 = cos(-5π/8)
    -0.70710678118f,         // W_16^6 = cos(-3π/4)
    -0.92387953251f          // W_16^7 = cos(-7π/8)
};
__constant__ float TWIDDLE_SIN_STAGE3[8] = {
    0.0f,                    // W_16^0 = sin(0)
    -0.38268343236f,         // W_16^1 = sin(-π/8)
    -0.70710678118f,         // W_16^2 = sin(-π/4) = -√2/2
    -0.92387953251f,         // W_16^3 = sin(-3π/8)
    -1.0f,                   // W_16^4 = sin(-π/2)
    -0.92387953251f,         // W_16^5 = sin(-5π/8)
    -0.70710678118f,         // W_16^6 = sin(-3π/4)
    -0.38268343236f          // W_16^7 = sin(-7π/8)
};

/**
 * @brief Get twiddle factor for specific stage and index
 * @param stage Butterfly stage (0-3)
 * @param k Index within stage
 * @param[out] cos_w Cosine component
 * @param[out] sin_w Sine component
 */
__device__ __forceinline__ void get_twiddle(int stage, int k, float& cos_w, float& sin_w) {
    switch(stage) {
        case 0:
            cos_w = TWIDDLE_COS_STAGE0[k];
            sin_w = TWIDDLE_SIN_STAGE0[k];
            break;
        case 1:
            cos_w = TWIDDLE_COS_STAGE1[k];
            sin_w = TWIDDLE_SIN_STAGE1[k];
            break;
        case 2:
            cos_w = TWIDDLE_COS_STAGE2[k];
            sin_w = TWIDDLE_SIN_STAGE2[k];
            break;
        case 3:
            cos_w = TWIDDLE_COS_STAGE3[k];
            sin_w = TWIDDLE_SIN_STAGE3[k];
            break;
    }
}

} // namespace CudaCalc

