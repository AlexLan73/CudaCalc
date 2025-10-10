/**
 * @file fft32_wmma_v2_ultra_kernel.cu
 * @brief ULTRA-OPTIMIZED FFT32 kernel (EXPERIMENT 2)
 * 
 * Inspired by AMGpuCuda ultra_fft_kernels.cu but adapted for COMPLEX FFT
 * 
 * Key differences from old project:
 * - OLD: REAL FFT (float data)
 * - NEW: COMPLEX FFT (cuComplex data)
 * 
 * Optimizations from old project:
 * - 1D thread indexing (simpler than 2D!)
 * - Dynamic shared memory
 * - Coalesced memory access
 * - Optimized butterfly pattern
 * 
 * Configuration:
 * - 1024 threads per block (1D)
 * - 32 FFT per block (32 threads per FFT)
 * - Thread ID = fftId * 32 + pointId
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>
#include <string>

// === CONSTANT MEMORY: Pre-computed twiddle factors for FFT32 ===
__constant__ float TWIDDLES_V2_32_REAL[32] = {
    1.000000f,   0.980785f,   0.923880f,   0.831470f,
    0.707107f,   0.555570f,   0.382683f,   0.195090f,
    0.000000f,  -0.195090f,  -0.382683f,  -0.555570f,
   -0.707107f,  -0.831470f,  -0.923880f,  -0.980785f,
   -1.000000f,  -0.980785f,  -0.923880f,  -0.831470f,
   -0.707107f,  -0.555570f,  -0.382683f,  -0.195090f,
   -0.000000f,   0.195090f,   0.382683f,   0.555570f,
    0.707107f,   0.831470f,   0.923880f,   0.980785f
};

__constant__ float TWIDDLES_V2_32_IMAG[32] = {
   -0.000000f,  -0.195090f,  -0.382683f,  -0.555570f,
   -0.707107f,  -0.831470f,  -0.923880f,  -0.980785f,
   -1.000000f,  -0.980785f,  -0.923880f,  -0.831470f,
   -0.707107f,  -0.555570f,  -0.382683f,  -0.195090f,
    0.000000f,   0.195090f,   0.382683f,   0.555570f,
    0.707107f,   0.831470f,   0.923880f,   0.980785f,
    1.000000f,   0.980785f,   0.923880f,   0.831470f,
    0.707107f,   0.555570f,   0.382683f,   0.195090f
};

__constant__ int BIT_REVERSED_V2_32[32] = {
     0, 16,  8, 24,  4, 20, 12, 28,
     2, 18, 10, 26,  6, 22, 14, 30,
     1, 17,  9, 25,  5, 21, 13, 29,
     3, 19, 11, 27,  7, 23, 15, 31
};

/**
 * @brief ULTRA-OPTIMIZED FFT32 kernel with 1D indexing (like old project)
 * 
 * Thread organization:
 * - 1024 threads = 32 FFT × 32 points
 * - threadId = fftId * 32 + pointId
 * - fftId = threadId / 32 (which FFT: 0..31)
 * - pointId = threadId % 32 (which point: 0..31)
 */
__global__ void fft32_v2_ultra_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    // === 1D THREAD INDEXING (simpler than 2D!) ===
    const int threadId = threadIdx.x;        // 0..1023
    const int blockId = blockIdx.x;
    
    const int fftId = threadId >> 5;         // threadId / 32 (bit shift faster!)
    const int pointId = threadId & 31;       // threadId % 32
    
    if (fftId >= 32) return;
    
    const int globalFFTId = (blockId << 5) + fftId;  // blockId * 32 + fftId
    if (globalFFTId >= num_windows) return;
    
    // === DYNAMIC SHARED MEMORY (like old project) ===
    // Each FFT gets 32 complex points
    extern __shared__ float2 shared_mem[];
    float2* fft_data = &shared_mem[fftId << 5];  // fftId * 32
    
    // === LOAD INPUT WITH BIT-REVERSAL ===
    const int input_idx = (globalFFTId << 5) + pointId;
    const int reversed_idx = BIT_REVERSED_V2_32[pointId];
    const cuComplex loaded = __ldg(&input[input_idx]);
    fft_data[reversed_idx] = make_float2(loaded.x, loaded.y);
    
    __syncthreads();
    
    // === BUTTERFLY FFT (5 stages for 32 points) ===
    
    // STAGE 0: step = 1, 16 pairs
    {
        if (pointId < 16) {
            const int idx1 = pointId << 1;       // pointId * 2
            const int idx2 = idx1 + 1;
            
            float2 a = fft_data[idx1];
            float2 b = fft_data[idx2];
            
            fft_data[idx1] = make_float2(a.x + b.x, a.y + b.y);
            fft_data[idx2] = make_float2(a.x - b.x, a.y - b.y);
        }
        __syncthreads();
    }
    
    // STAGE 1: step = 2, 8 groups of 4
    {
        if (pointId < 16) {
            const int group = pointId >> 2;      // pointId / 4
            const int in_group = pointId & 3;    // pointId % 4
            const int idx1 = (group << 2) + in_group;
            const int idx2 = idx1 + 2;
            
            float2 a = fft_data[idx1];
            float2 b = fft_data[idx2];
            
            // Twiddle
            const int tw_idx = in_group << 3;    // in_group * 8
            const float tw_re = TWIDDLES_V2_32_REAL[tw_idx];
            const float tw_im = TWIDDLES_V2_32_IMAG[tw_idx];
            
            float b_tw_re = b.x * tw_re - b.y * tw_im;
            float b_tw_im = b.x * tw_im + b.y * tw_re;
            
            fft_data[idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
            fft_data[idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
        }
        __syncthreads();
    }
    
    // STAGE 2: step = 4, 4 groups of 8
    {
        if (pointId < 16) {
            const int group = pointId >> 3;      // pointId / 8
            const int in_group = pointId & 7;    // pointId % 8
            const int idx1 = (group << 3) + in_group;
            const int idx2 = idx1 + 4;
            
            float2 a = fft_data[idx1];
            float2 b = fft_data[idx2];
            
            const int tw_idx = in_group << 2;    // in_group * 4
            const float tw_re = TWIDDLES_V2_32_REAL[tw_idx];
            const float tw_im = TWIDDLES_V2_32_IMAG[tw_idx];
            
            float b_tw_re = b.x * tw_re - b.y * tw_im;
            float b_tw_im = b.x * tw_im + b.y * tw_re;
            
            fft_data[idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
            fft_data[idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
        }
        __syncthreads();
    }
    
    // STAGE 3: step = 8, 2 groups of 16
    {
        if (pointId < 16) {
            const int group = pointId >> 4;      // pointId / 16
            const int in_group = pointId & 15;   // pointId % 16
            const int idx1 = (group << 4) + in_group;
            const int idx2 = idx1 + 8;
            
            float2 a = fft_data[idx1];
            float2 b = fft_data[idx2];
            
            const int tw_idx = in_group << 1;    // in_group * 2
            const float tw_re = TWIDDLES_V2_32_REAL[tw_idx];
            const float tw_im = TWIDDLES_V2_32_IMAG[tw_idx];
            
            float b_tw_re = b.x * tw_re - b.y * tw_im;
            float b_tw_im = b.x * tw_im + b.y * tw_re;
            
            fft_data[idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
            fft_data[idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
        }
        __syncthreads();
    }
    
    // STAGE 4: step = 16, 1 group of 32
    {
        if (pointId < 16) {
            const int idx1 = pointId;
            const int idx2 = pointId + 16;
            
            float2 a = fft_data[idx1];
            float2 b = fft_data[idx2];
            
            const float tw_re = TWIDDLES_V2_32_REAL[pointId];
            const float tw_im = TWIDDLES_V2_32_IMAG[pointId];
            
            float b_tw_re = b.x * tw_re - b.y * tw_im;
            float b_tw_im = b.x * tw_im + b.y * tw_re;
            
            fft_data[idx1] = make_float2(a.x + b_tw_re, a.y + b_tw_im);
            fft_data[idx2] = make_float2(a.x - b_tw_re, a.y - b_tw_im);
        }
        __syncthreads();
    }
    
    // === STORE OUTPUT WITH FFT SHIFT ===
    const int output_idx = (globalFFTId << 5) + pointId;
    const int shifted_p = (pointId < 16) ? (pointId + 16) : (pointId - 16);
    output[output_idx].x = fft_data[shifted_p].x;
    output[output_idx].y = fft_data[shifted_p].y;
}

// ==========================================
// KERNEL LAUNCHER (V2)
// ==========================================

void launch_fft32_v2_ultra_kernel(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    // EXPERIMENT 2: 1D blocks with dynamic shared memory
    const int fft_per_block = 32;
    const int threads_per_block = 1024;  // 32 FFT × 32 points
    const int num_blocks = (num_windows + fft_per_block - 1) / fft_per_block;
    
    // Dynamic shared memory: 32 FFT × 32 points × sizeof(float2)
    const size_t shared_mem_size = fft_per_block * 32 * sizeof(float2);
    
    fft32_v2_ultra_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        d_input, d_output, num_windows
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel V2 launch failed: ") + cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}

