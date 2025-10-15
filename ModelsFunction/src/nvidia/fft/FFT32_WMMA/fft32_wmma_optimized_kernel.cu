/**
 * @file fft32_wmma_optimized_kernel.cu
 * @brief CORRECT FFT32 GPU kernel
 * 
 * Based on working FFT16 algorithm!
 * One FFT per block, 32 threads per block
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace CudaCalc {

// Bit reverse for 5 bits
__device__ int bitReverse5(int x) {
    int result = 0;
    result |= (x & 1) << 4;
    result |= (x & 2) << 2;
    result |= (x & 4);
    result |= (x & 8) >> 2;
    result |= (x & 16) >> 4;
    return result;
}

/**
 * SIMPLE FFT32 kernel - one FFT per block
 * Block: 32 threads
 * Grid: num_windows blocks
 */
__global__ void fft32_simple_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int fft_id = blockIdx.x;
    const int tid = threadIdx.x;  // 0-31
    
    if (fft_id >= num_windows) return;
    
    __shared__ float2 data[32];
    
    // === STEP 1: Load with bit-reversal ===
    const int input_idx = fft_id * 32 + tid;
    const int reversed_tid = bitReverse5(tid);
    data[reversed_tid] = make_float2(input[input_idx].x, input[input_idx].y);
    __syncthreads();
    
    // === STEP 2: Butterfly stages (5 stages for FFT32) ===
    for (int stage = 0; stage < 5; ++stage) {
        const int m = 1 << (stage + 1);  // 2, 4, 8, 16, 32
        const int m2 = m / 2;             // 1, 2, 4,  8, 16
        
        // Each thread works on pairs in parallel
        const int k = (tid / m2) * m;  // Group start
        const int j = tid % m2;         // Position in group
        
        if (tid < 16) {  // Only 16 threads work (rest wait)
            const int idx1 = k + j;
            const int idx2 = idx1 + m2;
            
            // Compute twiddle: W = exp(-2πi*j/m)
            const float angle = -2.0f * M_PI * j / m;
            const float tw_cos = cosf(angle);
            const float tw_sin = sinf(angle);
            
            // Load values
            float2 u = data[idx1];
            float2 v = data[idx2];
            
            // t = W * v (complex multiply)
            float2 t;
            t.x = v.x * tw_cos - v.y * tw_sin;
            t.y = v.x * tw_sin + v.y * tw_cos;
            
            // Butterfly
            data[idx1] = make_float2(u.x + t.x, u.y + t.y);
            data[idx2] = make_float2(u.x - t.x, u.y - t.y);
        }
        __syncthreads();
    }
    
    // === STEP 3: Store result (NO shift!) ===
    const int output_idx = fft_id * 32 + tid;
    output[output_idx] = make_cuComplex(data[tid].x, data[tid].y);
}

// Host launcher (compatible with existing code)
void launch_fft32_wmma_optimized_kernel(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(32);  // 32 threads per block
    dim3 grid(num_windows);  // One block per FFT
    
    fft32_simple_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}

} // namespace CudaCalc

