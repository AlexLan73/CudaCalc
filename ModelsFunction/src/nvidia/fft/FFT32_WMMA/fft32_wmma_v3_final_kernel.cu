/**
 * @file fft32_wmma_v3_final_kernel.cu
 * @brief FFT32 FINAL VERSION - Based on successful FFT16_WMMA_Optimized!
 * 
 * COPIED from FFT16_WMMA_Optimized (which achieved 35% faster than target!)
 * Adapted for FFT32: 5 stages, 32 twiddles, 32 points
 * 
 * Key optimizations (same as FFT16):
 * - 2D blocks [64, 16] = 1024 threads BUT each thread processes 2 points!
 * - Constant memory twiddles
 * - __ldg() for read-only loads
 * - Minimal __syncthreads()
 * - Bit shifts for all arithmetic
 * - Shared memory padding
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>
#include <string>

namespace CudaCalc {

// === CONSTANT MEMORY: Twiddle factors for FFT32 ===
__constant__ float TWIDDLE_V3_REAL_32[32] = {
    1.000000f,   0.980785f,   0.923880f,   0.831470f,
    0.707107f,   0.555570f,   0.382683f,   0.195090f,
    0.000000f,  -0.195090f,  -0.382683f,  -0.555570f,
   -0.707107f,  -0.831470f,  -0.923880f,  -0.980785f,
   -1.000000f,  -0.980785f,  -0.923880f,  -0.831470f,
   -0.707107f,  -0.555570f,  -0.382683f,  -0.195090f,
   -0.000000f,   0.195090f,   0.382683f,   0.555570f,
    0.707107f,   0.831470f,   0.923880f,   0.980785f
};

__constant__ float TWIDDLE_V3_IMAG_32[32] = {
   -0.000000f,  -0.195090f,  -0.382683f,  -0.555570f,
   -0.707107f,  -0.831470f,  -0.923880f,  -0.980785f,
   -1.000000f,  -0.980785f,  -0.923880f,  -0.831470f,
   -0.707107f,  -0.555570f,  -0.382683f,  -0.195090f,
    0.000000f,   0.195090f,   0.382683f,   0.555570f,
    0.707107f,   0.831470f,   0.923880f,   0.980785f,
    1.000000f,   0.980785f,   0.923880f,   0.831470f,
    0.707107f,   0.555570f,   0.382683f,   0.195090f
};

__constant__ int BIT_REVERSED_V3_32[32] = {
     0, 16,  8, 24,  4, 20, 12, 28,
     2, 18, 10, 26,  6, 22, 14, 30,
     1, 17,  9, 25,  5, 21, 13, 29,
     3, 19, 11, 27,  7, 23, 15, 31
};

/**
 * @brief FFT32 V3 FINAL kernel - Based on FFT16 success!
 * 
 * Configuration (SAME as FFT16):
 * - Block: 2D [64, 16] = 1024 threads
 * - 64 FFT per block
 * - Each thread: handles 2 points (16*2 = 32)
 * - FP32 (no conversion overhead)
 * - Constant memory twiddles
 * - __ldg() for inputs
 * - Minimal synchronization
 */
__global__ void fft32_v3_final_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    // === 2D INDEXING: [64 FFT, 16 threads] ===
    const int x = threadIdx.x;  // 0-63: which FFT
    const int y = threadIdx.y;  // 0-15: which point (but we need 32!)
    const int global_fft_id = (blockIdx.x << 6) + x;  // blockIdx.x * 64 + x
    
    if (global_fft_id >= num_windows) return;
    
    // === SHARED MEMORY: [64 FFT][32 points + 2 padding] ===
    __shared__ float2 shmem[64][34];
    
    // === LOAD INPUT with BIT-REVERSAL (each thread loads 2 points) ===
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int point_idx = y + (p << 4);  // y + p*16 (0..31)
        const int input_idx = (global_fft_id << 5) + point_idx;  // *32
        const int reversed_idx = BIT_REVERSED_V3_32[point_idx];
        
        const cuComplex val = __ldg(&input[input_idx]);
        shmem[x][reversed_idx] = make_float2(val.x, val.y);
    }
    __syncthreads();
    
    // ===================================================================
    // BUTTERFLY: LINEAR UNROLL with register reuse (COPIED FROM FFT16!)
    // ===================================================================
    
    // STAGE 0: step = 1, 16 pairs
    // Each thread processes 2 butterflies
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int point_idx = y + (p << 4);
        if (point_idx < 16) {
            const int idx1 = point_idx << 1;
            const int idx2 = idx1 + 1;
            
            float2 a = shmem[x][idx1];
            float2 b = shmem[x][idx2];
            
            shmem[x][idx1] = make_float2(a.x + b.x, a.y + b.y);
            shmem[x][idx2] = make_float2(a.x - b.x, a.y - b.y);
        }
    }
    __syncthreads();
    
    // STAGE 1: step = 2, 8 groups of 4
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int point_idx = y + (p << 4);
        if (point_idx < 16) {
            const int group = point_idx >> 2;   // / 4
            const int pos = point_idx & 3;      // % 4
            const int idx1 = (group << 2) + pos;
            const int idx2 = idx1 + 2;
            
            float2 a = shmem[x][idx1];
            float2 b = shmem[x][idx2];
            
            // Twiddle: W_32^(pos*8)
            const int tw_idx = pos << 3;  // pos * 8
            const float tw_re = TWIDDLE_V3_REAL_32[tw_idx];
            const float tw_im = TWIDDLE_V3_IMAG_32[tw_idx];
            
            const float b_tw_r = b.x * tw_re - b.y * tw_im;
            const float b_tw_i = b.x * tw_im + b.y * tw_re;
            
            shmem[x][idx1] = make_float2(a.x + b_tw_r, a.y + b_tw_i);
            shmem[x][idx2] = make_float2(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // STAGE 2: step = 4, 4 groups of 8
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int point_idx = y + (p << 4);
        if (point_idx < 16) {
            const int group = point_idx >> 3;   // / 8
            const int pos = point_idx & 7;      // % 8
            const int idx1 = (group << 3) + pos;
            const int idx2 = idx1 + 4;
            
            float2 a = shmem[x][idx1];
            float2 b = shmem[x][idx2];
            
            // Twiddle: W_32^(pos*4)
            const int tw_idx = pos << 2;  // pos * 4
            const float tw_re = TWIDDLE_V3_REAL_32[tw_idx];
            const float tw_im = TWIDDLE_V3_IMAG_32[tw_idx];
            
            const float b_tw_r = b.x * tw_re - b.y * tw_im;
            const float b_tw_i = b.x * tw_im + b.y * tw_re;
            
            shmem[x][idx1] = make_float2(a.x + b_tw_r, a.y + b_tw_i);
            shmem[x][idx2] = make_float2(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // STAGE 3: step = 8, 2 groups of 16
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int point_idx = y + (p << 4);
        if (point_idx < 16) {
            const int group = point_idx >> 4;   // / 16
            const int pos = point_idx & 15;     // % 16
            const int idx1 = (group << 4) + pos;
            const int idx2 = idx1 + 8;
            
            float2 a = shmem[x][idx1];
            float2 b = shmem[x][idx2];
            
            // Twiddle: W_32^(pos*2)
            const int tw_idx = pos << 1;  // pos * 2
            const float tw_re = TWIDDLE_V3_REAL_32[tw_idx];
            const float tw_im = TWIDDLE_V3_IMAG_32[tw_idx];
            
            const float b_tw_r = b.x * tw_re - b.y * tw_im;
            const float b_tw_i = b.x * tw_im + b.y * tw_re;
            
            shmem[x][idx1] = make_float2(a.x + b_tw_r, a.y + b_tw_i);
            shmem[x][idx2] = make_float2(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // STAGE 4: step = 16, 1 group of 32
    // CRITICAL: Now we need ALL 32 points!
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int point_idx = y + (p << 4);  // 0..31
        const int idx1 = point_idx;
        const int idx2 = point_idx + 16;
        
        float2 a = shmem[x][idx1];
        float2 b = shmem[x][idx2];
        
        // Twiddle: W_32^point_idx
        const float tw_re = TWIDDLE_V3_REAL_32[point_idx];
        const float tw_im = TWIDDLE_V3_IMAG_32[point_idx];
        
        const float b_tw_r = b.x * tw_re - b.y * tw_im;
        const float b_tw_i = b.x * tw_im + b.y * tw_re;
        
        shmem[x][idx1] = make_float2(a.x + b_tw_r, a.y + b_tw_i);
        shmem[x][idx2] = make_float2(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // === STORE OUTPUT WITH FFT SHIFT ===
    #pragma unroll
    // === STORE OUTPUT (NO SHIFT!) ===
    for (int p = 0; p < 2; ++p) {
        const int point_idx = y + (p << 4);
        const int output_idx = (global_fft_id << 5) + point_idx;
        
        output[output_idx].x = shmem[x][point_idx].x;
        output[output_idx].y = shmem[x][point_idx].y;
    }
}

// ==========================================
// KERNEL LAUNCHER (V3)
// ==========================================

void launch_fft32_v3_final_kernel(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    // SAME configuration as FFT16 Optimized: [64 FFT][16 threads]
    const int fft_per_block = 64;
    const int num_blocks = (num_windows + fft_per_block - 1) / fft_per_block;
    dim3 block_dim(64, 16);  // 1024 threads total
    dim3 grid_dim(num_blocks);
    
    fft32_v3_final_kernel<<<grid_dim, block_dim>>>(d_input, d_output, num_windows);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA V3 kernel failed: ") + cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}

} // namespace CudaCalc

