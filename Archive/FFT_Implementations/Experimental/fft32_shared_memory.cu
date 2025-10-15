/**
 * @file fft32_shared_memory.cu
 * @brief FFT32 GPU kernel using shared memory
 * 
 * Точная копия рабочего fft32_simple_correct.cu алгоритма,
 * адаптированная для shared memory
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// Pre-computed twiddle factors for FFT32 (копируем из рабочего алгоритма)
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

// Bit reverse for 5 bits (копируем из рабочего алгоритма)
__device__ int bitReverse5_shared(int x) {
    int result = 0;
    result |= (x & 1) << 4;
    result |= (x & 2) << 2;
    result |= (x & 4);
    result |= (x & 8) >> 2;
    result |= (x & 16) >> 4;
    return result;
}

/**
 * FFT32 Shared Memory kernel - ТОЧНАЯ копия рабочего алгоритма
 * Block: 32 threads
 * Grid: num_windows blocks
 */
__global__ void fft32_shared_memory_kernel(
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
    const int reversed_tid = bitReverse5_shared(tid);
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
            
            // Get twiddle from pre-computed table
            // W = W_32^((j * 32) / m)
            const int twiddle_idx = (j * 32) / m;
            const float tw_cos = TWIDDLE_32_COS[twiddle_idx];
            const float tw_sin = TWIDDLE_32_SIN[twiddle_idx];
            
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

// Host launcher
extern "C" void launch_fft32_shared_memory(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(32);  // 32 threads per block
    dim3 grid(num_windows);  // One block per FFT
    
    fft32_shared_memory_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}


