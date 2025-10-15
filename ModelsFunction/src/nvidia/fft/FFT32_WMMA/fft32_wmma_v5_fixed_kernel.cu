/**
 * @file fft32_wmma_v5_fixed_kernel.cu
 * @brief FFT32 V5 FIXED - Corrected twiddle indices!
 * 
 * Based on V1 but with CORRECTED twiddle factor application
 * Following EXACT pattern from FFT16 (which works with 0.45% error)
 * 
 * Configuration: [32, 32] 2D blocks (BEST performance from experiments)
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

// === CONSTANT MEMORY: Twiddle factors W_32^k ===
__constant__ float TWIDDLES_V5_32_REAL[16] = {
    1.000000f,   0.980785f,   0.923880f,   0.831470f,
    0.707107f,   0.555570f,   0.382683f,   0.195090f,
    0.000000f,  -0.195090f,  -0.382683f,  -0.555570f,
   -0.707107f,  -0.831470f,  -0.923880f,  -0.980785f
};

__constant__ float TWIDDLES_V5_32_IMAG[16] = {
   -0.000000f,  -0.195090f,  -0.382683f,  -0.555570f,
   -0.707107f,  -0.831470f,  -0.923880f,  -0.980785f,
   -1.000000f,  -0.980785f,  -0.923880f,  -0.831470f,
   -0.707107f,  -0.555570f,  -0.382683f,  -0.195090f
};

__constant__ int BIT_REVERSED_V5_32[32] = {
     0, 16,  8, 24,  4, 20, 12, 28,
     2, 18, 10, 26,  6, 22, 14, 30,
     1, 17,  9, 25,  5, 21, 13, 29,
     3, 19, 11, 27,  7, 23, 15, 31
};

__global__ void fft32_v5_fixed_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int block_fft_id = threadIdx.x;  // 0..31
    const int point_id = threadIdx.y;      // 0..31
    const int global_fft_id = (blockIdx.x << 5) + block_fft_id;

    if (global_fft_id >= num_windows) return;

    __shared__ float2 shmem[32][34];

    // Load with bit-reversal
    const int input_idx = (global_fft_id << 5) + point_id;
    const int reversed_idx = BIT_REVERSED_V5_32[point_id];
    float2 loaded_val = make_float2(__ldg(&input[input_idx].x), __ldg(&input[input_idx].y));
    shmem[block_fft_id][reversed_idx] = loaded_val;
    __syncthreads();

    // === CORRECTED BUTTERFLY STAGES ===
    
    // STAGE 0: step=1 (16 pairs of 2-point FFTs)
    {
        const int pair_id = point_id >> 1;
        const int idx1 = pair_id << 1;
        const int idx2 = idx1 + 1;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        shmem[block_fft_id][idx1] = make_float2(a.x + b.x, a.y + b.y);
        shmem[block_fft_id][idx2] = make_float2(a.x - b.x, a.y - b.y);
        __syncthreads();
    }

    // STAGE 1: step=2 (8 groups of 4-point FFTs)
    {
        const int group_id = point_id >> 2;      // / 4
        const int in_group = point_id & 3;       // % 4
        const int idx1 = (group_id << 2) + in_group;
        const int idx2 = idx1 + 2;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        // CORRECTED: W_32^(in_group * 8) for step=2
        // Pattern from FFT16: step=2 uses pos*4, so FFT32 step=2 uses pos*8
        const int tw_idx = in_group << 3;  // in_group * 8
        const float tw_re = TWIDDLES_V5_32_REAL[tw_idx & 15];
        const float tw_im = TWIDDLES_V5_32_IMAG[tw_idx & 15];

        float b_tw_re = b.x * tw_re - b.y * tw_im;
        float b_tw_im = b.x * tw_im + b.y * tw_re;

        shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
        shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
        __syncthreads();
    }

    // STAGE 2: step=4 (4 groups of 8-point FFTs)
    {
        const int group_id = point_id >> 3;      // / 8
        const int in_group = point_id & 7;       // % 8
        const int idx1 = (group_id << 3) + in_group;
        const int idx2 = idx1 + 4;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        // W_32^(in_group * 4)
        const int tw_idx = in_group << 2;  // in_group * 4
        const float tw_re = TWIDDLES_V5_32_REAL[tw_idx & 15];
        const float tw_im = TWIDDLES_V5_32_IMAG[tw_idx & 15];

        float b_tw_re = b.x * tw_re - b.y * tw_im;
        float b_tw_im = b.x * tw_im + b.y * tw_re;

        shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
        shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
        __syncthreads();
    }

    // STAGE 3: step=8 (2 groups of 16-point FFTs)
    {
        const int group_id = point_id >> 4;      // / 16
        const int in_group = point_id & 15;      // % 16
        const int idx1 = (group_id << 4) + in_group;
        const int idx2 = idx1 + 8;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        // W_32^(in_group * 2)
        const int tw_idx = in_group << 1;  // in_group * 2
        const float tw_re = TWIDDLES_V5_32_REAL[tw_idx & 15];
        const float tw_im = TWIDDLES_V5_32_IMAG[tw_idx & 15];

        float b_tw_re = b.x * tw_re - b.y * tw_im;
        float b_tw_im = b.x * tw_im + b.y * tw_re;

        shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
        shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
        __syncthreads();
    }

    // STAGE 4: step=16 (1 group of 32-point FFT)
    {
        const int idx1 = point_id;
        const int idx2 = point_id + 16;

        float2 a = shmem[block_fft_id][idx1];
        float2 b = shmem[block_fft_id][idx2];

        // W_32^point_id (direct, for final stage)
        const float tw_re = TWIDDLES_V5_32_REAL[point_id & 15];
        const float tw_im = TWIDDLES_V5_32_IMAG[point_id & 15];

        float b_tw_re = b.x * tw_re - b.y * tw_im;
        float b_tw_im = b.x * tw_im + b.y * tw_re;

        shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
        shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
        __syncthreads();
    }

    // === STORE OUTPUT (NO SHIFT!) ===
    const int output_idx = (global_fft_id << 5) + point_id;
    output[output_idx].x = shmem[block_fft_id][point_id].x;
    output[output_idx].y = shmem[block_fft_id][point_id].y;
}

namespace CudaCalc {

void launch_fft32_v5_fixed(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    const int fft_per_block = 32;
    const int num_blocks = (num_windows + fft_per_block - 1) / fft_per_block;
    dim3 block_dim(32, 32);
    dim3 grid_dim(num_blocks);

    fft32_v5_fixed_kernel<<<grid_dim, block_dim>>>(d_input, d_output, num_windows);
    
    cudaDeviceSynchronize();
}

} // namespace CudaCalc


