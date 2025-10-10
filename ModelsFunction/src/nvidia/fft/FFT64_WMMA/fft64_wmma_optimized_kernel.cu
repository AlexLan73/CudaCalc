/**
 * @file fft64_wmma_optimized_kernel.cu
 * @brief FFT64 Optimized - Based on FFT16 winner pattern!
 * 
 * Configuration: [64, 16] 2D blocks = 1024 threads (same as FFT16!)
 * Strategy: Each thread handles 4 points (64/16 = 4) with loops
 * 
 * Based on FFT16_WMMA_Optimized which achieved 0.00512ms!
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

// === CONSTANT MEMORY: Twiddle factors W_64^k ===
__constant__ float TWIDDLES_64_REAL[32] = {
    1.000000f,   0.995185f,   0.980785f,   0.956940f,
    0.923880f,   0.881921f,   0.831470f,   0.773010f,
    0.707107f,   0.634393f,   0.555570f,   0.471397f,
    0.382683f,   0.290285f,   0.195090f,   0.098017f,
    0.000000f,  -0.098017f,  -0.195090f,  -0.290285f,
   -0.382683f,  -0.471397f,  -0.555570f,  -0.634393f,
   -0.707107f,  -0.773010f,  -0.831470f,  -0.881921f,
   -0.923880f,  -0.956940f,  -0.980785f,  -0.995185f
};

__constant__ float TWIDDLES_64_IMAG[32] = {
   -0.000000f,  -0.098017f,  -0.195090f,  -0.290285f,
   -0.382683f,  -0.471397f,  -0.555570f,  -0.634393f,
   -0.707107f,  -0.773010f,  -0.831470f,  -0.881921f,
   -0.923880f,  -0.956940f,  -0.980785f,  -0.995185f,
   -1.000000f,  -0.995185f,  -0.980785f,  -0.956940f,
   -0.923880f,  -0.881921f,  -0.831470f,  -0.773010f,
   -0.707107f,  -0.634393f,  -0.555570f,  -0.471397f,
   -0.382683f,  -0.290285f,  -0.195090f,  -0.098017f
};

__constant__ int BIT_REVERSED_64[64] = {
     0, 32, 16, 48,  8, 40, 24, 56,
     4, 36, 20, 52, 12, 44, 28, 60,
     2, 34, 18, 50, 10, 42, 26, 58,
     6, 38, 22, 54, 14, 46, 30, 62,
     1, 33, 17, 49,  9, 41, 25, 57,
     5, 37, 21, 53, 13, 45, 29, 61,
     3, 35, 19, 51, 11, 43, 27, 59,
     7, 39, 23, 55, 15, 47, 31, 63
};

__global__ void fft64_optimized_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int block_fft_id = threadIdx.x;  // 0..63
    const int y = threadIdx.y;             // 0..15
    const int global_fft_id = (blockIdx.x << 6) + block_fft_id;  // blockIdx.x * 64

    if (global_fft_id >= num_windows) return;

    __shared__ float2 shmem[64][66];  // Padding 66 = 64 + 2

    // === LOAD INPUT WITH BIT-REVERSAL (each thread loads 4 points) ===
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int point_id = y + i * 16;  // 0-15, 16-31, 32-47, 48-63
        const int input_idx = (global_fft_id << 6) + point_id;
        const int reversed_idx = BIT_REVERSED_64[point_id];
        float2 loaded_val = make_float2(__ldg(&input[input_idx].x), __ldg(&input[input_idx].y));
        shmem[block_fft_id][reversed_idx] = loaded_val;
    }
    __syncthreads();

    // === 6 BUTTERFLY STAGES (each thread handles 4 points) ===
    
    // STAGE 0: step=1 (32 pairs of 2-point FFTs)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int point_id = y + i * 16;
        const int pair_id = point_id >> 1;
        const int idx1 = pair_id << 1;
        const int idx2 = idx1 + 1;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        shmem[block_fft_id][idx1] = make_float2(a.x + b.x, a.y + b.y);
        shmem[block_fft_id][idx2] = make_float2(a.x - b.x, a.y - b.y);
    }
    __syncthreads();

    // STAGE 1: step=2 (16 groups of 4-point FFTs)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int point_id = y + i * 16;
        const int group_id = point_id >> 2;
        const int in_group = point_id & 3;
        const int idx1 = (group_id << 2) + in_group;
        const int idx2 = idx1 + 2;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        const int tw_idx = (in_group << 4) & 31;  // in_group * 16
        const float tw_re = TWIDDLES_64_REAL[tw_idx];
        const float tw_im = TWIDDLES_64_IMAG[tw_idx];

        float b_tw_re = b.x * tw_re - b.y * tw_im;
        float b_tw_im = b.x * tw_im + b.y * tw_re;

        shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
        shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
    }
    __syncthreads();

    // STAGE 2: step=4 (8 groups of 8-point FFTs)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int point_id = y + i * 16;
        const int group_id = point_id >> 3;
        const int in_group = point_id & 7;
        const int idx1 = (group_id << 3) + in_group;
        const int idx2 = idx1 + 4;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        const int tw_idx = (in_group << 3) & 31;  // in_group * 8
        const float tw_re = TWIDDLES_64_REAL[tw_idx];
        const float tw_im = TWIDDLES_64_IMAG[tw_idx];

        float b_tw_re = b.x * tw_re - b.y * tw_im;
        float b_tw_im = b.x * tw_im + b.y * tw_re;

        shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
        shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
    }
    __syncthreads();

    // STAGE 3: step=8 (4 groups of 16-point FFTs)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int point_id = y + i * 16;
        const int group_id = point_id >> 4;
        const int in_group = point_id & 15;
        const int idx1 = (group_id << 4) + in_group;
        const int idx2 = idx1 + 8;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        const int tw_idx = (in_group << 2) & 31;  // in_group * 4
        const float tw_re = TWIDDLES_64_REAL[tw_idx];
        const float tw_im = TWIDDLES_64_IMAG[tw_idx];

        float b_tw_re = b.x * tw_re - b.y * tw_im;
        float b_tw_im = b.x * tw_im + b.y * tw_re;

        shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
        shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
    }
    __syncthreads();

    // STAGE 4: step=16 (2 groups of 32-point FFTs)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int point_id = y + i * 16;
        const int group_id = point_id >> 5;
        const int in_group = point_id & 31;
        const int idx1 = (group_id << 5) + in_group;
        const int idx2 = idx1 + 16;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        const int tw_idx = (in_group << 1) & 31;  // in_group * 2
        const float tw_re = TWIDDLES_64_REAL[tw_idx];
        const float tw_im = TWIDDLES_64_IMAG[tw_idx];

        float b_tw_re = b.x * tw_re - b.y * tw_im;
        float b_tw_im = b.x * tw_im + b.y * tw_re;

        shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
        shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
    }
    __syncthreads();

    // STAGE 5: step=32 (1 group of 64-point FFT)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int point_id = y + i * 16;
        const int idx1 = point_id;
        const int idx2 = point_id + 32;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        const int tw_idx = point_id & 31;  // Direct index
        const float tw_re = TWIDDLES_64_REAL[tw_idx];
        const float tw_im = TWIDDLES_64_IMAG[tw_idx];

        float b_tw_re = b.x * tw_re - b.y * tw_im;
        float b_tw_im = b.x * tw_im + b.y * tw_re;

        shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
        shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
    }
    __syncthreads();

    // === STORE OUTPUT WITH FFT SHIFT (each thread writes 4 points) ===
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int point_id = y + i * 16;
        const int output_idx = (global_fft_id << 6) + point_id;
        const int shifted_p = (point_id < 32) ? (point_id + 32) : (point_id - 32);
        output[output_idx].x = shmem[block_fft_id][shifted_p].x;
        output[output_idx].y = shmem[block_fft_id][shifted_p].y;
    }
}

namespace CudaCalc {

void launch_fft64_optimized(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    const int fft_per_block = 64;
    const int num_blocks = (num_windows + fft_per_block - 1) / fft_per_block;
    dim3 block_dim(64, 16);
    dim3 grid_dim(num_blocks);

    fft64_optimized_kernel<<<grid_dim, block_dim>>>(d_input, d_output, num_windows);
    
    cudaDeviceSynchronize();
}

} // namespace CudaCalc

