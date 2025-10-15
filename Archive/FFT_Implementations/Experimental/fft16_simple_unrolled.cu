/**
 * @file fft16_simple_unrolled.cu
 * @brief CORRECT FFT16 with LINEAR UNROLL (no loops!)
 * 
 * Based on fft16_simple_correct_WITH_LOOP.cu
 * Butterfly stages unrolled for speed
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

// Pre-computed twiddle factors for FFT16: W_16^k = exp(-2πi*k/16)
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
 * FFT16 kernel with LINEAR UNROLL
 * Block: 16 threads
 * Grid: num_windows blocks
 */
__global__ void fft16_unrolled_kernel(
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
    
    // ===================================================================
    // BUTTERFLY STAGES - LINEAR UNROLL (no loops!)
    // ===================================================================
    
    // ========================================
    // STAGE 0: m=2, m2=1
    // ========================================
    if (tid < 8) {
        const int m2 = 1;
        const int k = (tid / m2) * 2;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        // twiddle_idx = (j * 16) / 2 = j * 8
        const int twiddle_idx = j * 8;
        const float tw_cos = TWIDDLE_16_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_16_SIN[twiddle_idx];
        
        float2 u = data[idx1];
        float2 v = data[idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        data[idx1] = make_float2(u.x + t.x, u.y + t.y);
        data[idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // ========================================
    // STAGE 1: m=4, m2=2
    // ========================================
    if (tid < 8) {
        const int m2 = 2;
        const int k = (tid / m2) * 4;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        // twiddle_idx = (j * 16) / 4 = j * 4
        const int twiddle_idx = j * 4;
        const float tw_cos = TWIDDLE_16_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_16_SIN[twiddle_idx];
        
        float2 u = data[idx1];
        float2 v = data[idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        data[idx1] = make_float2(u.x + t.x, u.y + t.y);
        data[idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // ========================================
    // STAGE 2: m=8, m2=4
    // ========================================
    if (tid < 8) {
        const int m2 = 4;
        const int k = (tid / m2) * 8;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        // twiddle_idx = (j * 16) / 8 = j * 2
        const int twiddle_idx = j * 2;
        const float tw_cos = TWIDDLE_16_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_16_SIN[twiddle_idx];
        
        float2 u = data[idx1];
        float2 v = data[idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        data[idx1] = make_float2(u.x + t.x, u.y + t.y);
        data[idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // ========================================
    // STAGE 3: m=16, m2=8 (final stage)
    // ========================================
    if (tid < 8) {
        const int m2 = 8;
        const int k = 0;  // Only one group
        const int j = tid;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        // twiddle_idx = (j * 16) / 16 = j
        const int twiddle_idx = j;
        const float tw_cos = TWIDDLE_16_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_16_SIN[twiddle_idx];
        
        float2 u = data[idx1];
        float2 v = data[idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        data[idx1] = make_float2(u.x + t.x, u.y + t.y);
        data[idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // === STEP 3: Store result (NO shift!) ===
    const int output_idx = fft_id * 16 + tid;
    output[output_idx] = make_cuComplex(data[tid].x, data[tid].y);
}

// Host launcher
extern "C" void launch_fft16_unrolled(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(16);
    dim3 grid(num_windows);
    
    fft16_unrolled_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}



