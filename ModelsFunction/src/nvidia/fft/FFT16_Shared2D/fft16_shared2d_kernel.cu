/**
 * @file fft16_shared2d_kernel.cu
 * @brief CUDA kernel for FFT16 with 2D Shared Memory
 * 
 * LINEAR UNROLL of 4 butterfly stages for maximum speed!
 * NO for loops in critical path.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace CudaCalc {

/**
 * @brief FFT16 kernel with LINEAR unroll (4 stages explicitly written)
 * 
 * Configuration:
 * - Block: 1024 threads = 64 FFT × 16 threads each
 * - Shared memory: [64 FFTs][16 points] 2D array
 * - Linear unroll: 4 butterfly stages without for loop
 * - FFT shift: applied in kernel at output
 * 
 * @param input Input data (num_windows × 16 complex points)
 * @param output Output data (num_windows × 16 complex points, shifted)
 * @param num_windows Number of FFT windows to process
 */
__global__ void fft16_shared2d_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    // === THREAD CONFIGURATION ===
    // 1024 threads per block = 64 FFT × 16 threads
    const int block_fft_id = threadIdx.x / 16;  // 0..63: which FFT in block
    const int point_id = threadIdx.x % 16;      // 0..15: which point in FFT
    const int global_fft_id = blockIdx.x * 64 + block_fft_id;
    
    if (global_fft_id >= num_windows) return;
    
    // === SHARED MEMORY: 2D ARRAY [64 FFTs][16 points] ===
    __shared__ float2 shmem[64][16];
    
    // === LOAD INPUT TO SHARED MEMORY WITH BIT-REVERSAL ===
    // For FFT16, bit-reversal permutation (4 bits):
    // 0→0, 1→8, 2→4, 3→12, 4→2, 5→10, 6→6, 7→14, 8→1, 9→9, 10→5, 11→13, 12→3, 13→11, 14→7, 15→15
    const int bit_reversed[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    
    const int input_idx = global_fft_id * 16 + point_id;
    const int reversed_idx = bit_reversed[point_id];
    shmem[block_fft_id][reversed_idx] = make_float2(input[input_idx].x, input[input_idx].y);
    
    __syncthreads();
    
    // ===================================================================
    // BUTTERFLY FFT: LINEAR UNROLL OF 4 STAGES (NO FOR LOOP!)
    // ===================================================================
    
    // -------------------------------------------------------------------
    // STAGE 0: step = 1, group_size = 2
    // -------------------------------------------------------------------
    {
        if (point_id < 8) {
            const int idx1 = point_id * 2;
            const int idx2 = idx1 + 1;
            
            float2 a = shmem[block_fft_id][idx1];
            float2 b = shmem[block_fft_id][idx2];
            
            // Twiddle for stage 0: W_2^k where k is position in pair
            // idx1: k=0, W_2^0 = 1
            // idx2: k=1, W_2^1 = exp(-i*π) = -1
            // So twiddle for idx2 is just -1 (no complex multiply needed!)
            
            // Butterfly: a + b*W and a - b*W, where W=1 for stage 0
            // Simplifies to: a + b and a - b
            shmem[block_fft_id][idx1] = make_float2(a.x + b.x, a.y + b.y);
            shmem[block_fft_id][idx2] = make_float2(a.x - b.x, a.y - b.y);
        }
        __syncthreads();
    }
    
    // -------------------------------------------------------------------
    // STAGE 1: step = 2, group_size = 4
    // -------------------------------------------------------------------
    {
        if (point_id < 8) {
            const int group = point_id / 2;
            const int pos = point_id % 2;
            const int idx1 = group * 4 + pos;
            const int idx2 = idx1 + 2;
            
            float2 a = shmem[block_fft_id][idx1];
            float2 b = shmem[block_fft_id][idx2];
            
            // Twiddle: W_4^k = exp(-i * π * k / 2)
            const float angle = -M_PI * pos / 2.0f;
            const float cos_w = cosf(angle);
            const float sin_w = sinf(angle);
            
            const float b_tw_real = b.x * cos_w - b.y * sin_w;
            const float b_tw_imag = b.x * sin_w + b.y * cos_w;
            
            shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_real, a.y + b_tw_imag);
            shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_real, a.y - b_tw_imag);
        }
        __syncthreads();
    }
    
    // -------------------------------------------------------------------
    // STAGE 2: step = 4, group_size = 8
    // -------------------------------------------------------------------
    {
        if (point_id < 8) {
            const int group = point_id / 4;
            const int pos = point_id % 4;
            const int idx1 = group * 8 + pos;
            const int idx2 = idx1 + 4;
            
            float2 a = shmem[block_fft_id][idx1];
            float2 b = shmem[block_fft_id][idx2];
            
            // Twiddle: W_8^k = exp(-i * π * k / 4)
            const float angle = -M_PI * pos / 4.0f;
            const float cos_w = cosf(angle);
            const float sin_w = sinf(angle);
            
            const float b_tw_real = b.x * cos_w - b.y * sin_w;
            const float b_tw_imag = b.x * sin_w + b.y * cos_w;
            
            shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_real, a.y + b_tw_imag);
            shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_real, a.y - b_tw_imag);
        }
        __syncthreads();
    }
    
    // -------------------------------------------------------------------
    // STAGE 3: step = 8, group_size = 16 (FINAL STAGE)
    // -------------------------------------------------------------------
    {
        if (point_id < 8) {
            const int idx1 = point_id;
            const int idx2 = idx1 + 8;
            
            float2 a = shmem[block_fft_id][idx1];
            float2 b = shmem[block_fft_id][idx2];
            
            // Twiddle: W_16^k = exp(-i * π * k / 8)
            const float angle = -M_PI * point_id / 8.0f;
            const float cos_w = cosf(angle);
            const float sin_w = sinf(angle);
            
            const float b_tw_real = b.x * cos_w - b.y * sin_w;
            const float b_tw_imag = b.x * sin_w + b.y * cos_w;
            
            shmem[block_fft_id][idx1] = make_float2(a.x + b_tw_real, a.y + b_tw_imag);
            shmem[block_fft_id][idx2] = make_float2(a.x - b_tw_real, a.y - b_tw_imag);
        }
        __syncthreads();
    }
    
    // ===================================================================
    // FFT SHIFT: Rearrange output to [-8, -7, ..., -1, DC, 1, ..., 7]
    // ===================================================================
    // Standard FFT order: [DC, 1, 2, ..., 7, 8, -7, -6, ..., -1]
    // After shift:        [-8, -7, -6, ..., -1, DC, 1, 2, ..., 7]
    
    int shifted_idx;
    if (point_id < 8) {
        // DC and positive frequencies: 0→8, 1→9, ..., 7→15
        shifted_idx = point_id + 8;
    } else {
        // Negative frequencies: 8→0, 9→1, ..., 15→7
        shifted_idx = point_id - 8;
    }
    
    // === STORE TO GLOBAL MEMORY WITH SHIFT ===
    const int output_idx = global_fft_id * 16 + shifted_idx;
    const float2 result = shmem[block_fft_id][point_id];
    output[output_idx] = make_cuComplex(result.x, result.y);
}

// Host wrapper function
void launch_fft16_shared2d(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows,
    cudaStream_t stream = 0
) {
    // Launch configuration
    const int num_blocks = (num_windows + 63) / 64;  // Ceiling division
    const int threads_per_block = 1024;               // 64 FFT × 16 threads
    const size_t shared_mem_size = 64 * 16 * sizeof(float2);  // 8192 bytes
    
    // Launch kernel
    fft16_shared2d_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
        d_input, d_output, num_windows
    );
}

} // namespace CudaCalc

