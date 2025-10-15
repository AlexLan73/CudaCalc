/**
 * @file fft16_simple_correct.cu
 * @brief SIMPLE and CORRECT FFT16 GPU kernel
 * 
 * Now with pre-computed twiddle tables!
 * One FFT per block, 16 threads per block
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// Pre-computed twiddle factors for FFT16: W_16^k = exp(-2πi*k/16)
// We need k = 0..7 (half table due to symmetry)
__constant__ float TWIDDLE_16_COS[8] = {
    1.000000f,   // k=0
    0.923880f,   // k=1
    0.707107f,   // k=2
    0.382683f,   // k=3
    0.000000f,   // k=4
   -0.382683f,   // k=5
   -0.707107f,   // k=6
   -0.923880f    // k=7
};

__constant__ float TWIDDLE_16_SIN[8] = {
    0.000000f,   // k=0
   -0.382683f,   // k=1
   -0.707107f,   // k=2
   -0.923880f,   // k=3
   -1.000000f,   // k=4
   -0.923880f,   // k=5
   -0.707107f,   // k=6
   -0.382683f    // k=7
};

// Bit reverse for 4 bits
__device__ int bitReverse4(int x) {
    int result = 0;
    result |= (x & 1) << 3;
    result |= (x & 2) << 1;
    result |= (x & 4) >> 1;
    result |= (x & 8) >> 3;
    return result;
}

/**
 * SIMPLE FFT16 kernel - one FFT per block
 * Block: 16 threads
 * Grid: num_windows blocks
 */
__global__ void fft16_simple_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int fft_id = blockIdx.x;
    const int tid = threadIdx.x;  // 0-15
    
    if (fft_id >= num_windows) return;
    
    __shared__ float2 data[16];
    
    // === STEP 1: Load with bit-reversal ===
    const int input_idx = fft_id * 16 + tid;
    const int reversed_tid = bitReverse4(tid);
    data[reversed_tid] = make_float2(input[input_idx].x, input[input_idx].y);
    __syncthreads();
    
    // === STEP 2: Butterfly stages (4 stages for FFT16) ===
    for (int stage = 0; stage < 4; ++stage) {
        const int m = 1 << (stage + 1);  // 2, 4, 8, 16
        const int m2 = m / 2;             // 1, 2, 4, 8
        
        // Each thread works on pairs in parallel
        const int k = (tid / m2) * m;  // Group start
        const int j = tid % m2;         // Position in group
        
        if (tid < 8) {  // Only 8 threads work (rest wait)
            const int idx1 = k + j;
            const int idx2 = idx1 + m2;
            
            // Get twiddle from pre-computed table
            // W = W_16^((j * 16) / m)
            const int twiddle_idx = (j * 16) / m;
            const float tw_cos = TWIDDLE_16_COS[twiddle_idx];
            const float tw_sin = TWIDDLE_16_SIN[twiddle_idx];
            
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
    const int output_idx = fft_id * 16 + tid;
    output[output_idx] = make_cuComplex(data[tid].x, data[tid].y);
}

// Host launcher
extern "C" void launch_fft16_simple(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(16);  // 16 threads per block
    dim3 grid(num_windows);  // One block per FFT
    
    fft16_simple_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}

