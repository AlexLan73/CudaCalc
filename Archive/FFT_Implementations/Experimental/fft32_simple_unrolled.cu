/**
 * @file fft32_simple_unrolled.cu
 * @brief CORRECT FFT32 with LINEAR UNROLL (no loops!)
 * 
 * Based on fft32_simple_correct_WITH_LOOP.cu
 * Butterfly stages unrolled for speed
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

// Pre-computed twiddle factors for FFT32: W_32^k = exp(-2πi*k/32)
__constant__ float TWIDDLE_32_COS[16] = {
    1.000000f,   // k=0
    0.980785f,   // k=1
    0.923880f,   // k=2
    0.831470f,   // k=3
    0.707107f,   // k=4
    0.555570f,   // k=5
    0.382683f,   // k=6
    0.195090f,   // k=7
    0.000000f,   // k=8
   -0.195090f,   // k=9
   -0.382683f,   // k=10
   -0.555570f,   // k=11
   -0.707107f,   // k=12
   -0.831470f,   // k=13
   -0.923880f,   // k=14
   -0.980785f    // k=15
};

__constant__ float TWIDDLE_32_SIN[16] = {
    0.000000f,   // k=0
   -0.195090f,   // k=1
   -0.382683f,   // k=2
   -0.555570f,   // k=3
   -0.707107f,   // k=4
   -0.831470f,   // k=5
   -0.923880f,   // k=6
   -0.980785f,   // k=7
   -1.000000f,   // k=8
   -0.980785f,   // k=9
   -0.923880f,   // k=10
   -0.831470f,   // k=11
   -0.707107f,   // k=12
   -0.555570f,   // k=13
   -0.382683f,   // k=14
   -0.195090f    // k=15
};

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
 * FFT32 kernel with LINEAR UNROLL
 * Block: 32 threads
 * Grid: num_windows blocks
 */
__global__ void fft32_unrolled_kernel(
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
    
    // ===================================================================
    // BUTTERFLY STAGES - LINEAR UNROLL (no loops!)
    // ===================================================================
    
    // ========================================
    // STAGE 0: m=2, m2=1
    // ========================================
    if (tid < 16) {
        const int m2 = 1;
        const int k = (tid / m2) * 2;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        // twiddle_idx = (j * 32) / 2 = j * 16
        const int twiddle_idx = j * 16;  // Will be 0 always for stage 0
        const float tw_cos = TWIDDLE_32_COS[twiddle_idx & 15];
        const float tw_sin = TWIDDLE_32_SIN[twiddle_idx & 15];
        
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
    if (tid < 16) {
        const int m2 = 2;
        const int k = (tid / m2) * 4;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        // twiddle_idx = (j * 32) / 4 = j * 8
        const int twiddle_idx = j * 8;
        const float tw_cos = TWIDDLE_32_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_32_SIN[twiddle_idx];
        
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
    if (tid < 16) {
        const int m2 = 4;
        const int k = (tid / m2) * 8;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        // twiddle_idx = (j * 32) / 8 = j * 4
        const int twiddle_idx = j * 4;
        const float tw_cos = TWIDDLE_32_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_32_SIN[twiddle_idx];
        
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
    // STAGE 3: m=16, m2=8
    // ========================================
    if (tid < 16) {
        const int m2 = 8;
        const int k = (tid / m2) * 16;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        // twiddle_idx = (j * 32) / 16 = j * 2
        const int twiddle_idx = j * 2;
        const float tw_cos = TWIDDLE_32_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_32_SIN[twiddle_idx];
        
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
    // STAGE 4: m=32, m2=16 (final stage)
    // ========================================
    if (tid < 16) {
        const int m2 = 16;
        const int k = 0;  // Only one group
        const int j = tid;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        // twiddle_idx = (j * 32) / 32 = j
        const int twiddle_idx = j;
        const float tw_cos = TWIDDLE_32_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_32_SIN[twiddle_idx];
        
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
    const int output_idx = fft_id * 32 + tid;
    output[output_idx] = make_cuComplex(data[tid].x, data[tid].y);
}

// Host launcher
extern "C" void launch_fft32_unrolled(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(32);
    dim3 grid(num_windows);
    
    fft32_unrolled_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}



