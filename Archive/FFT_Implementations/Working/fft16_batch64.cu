/**
 * @file fft16_batch64.cu
 * @brief FFT16 with 64 butterflies per block
 * 
 * Block: [64, 16] = 1024 threads
 * 64 FFT16 processed in parallel per block
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

// Pre-computed twiddle factors for FFT16
__constant__ float TWIDDLE_16_COS[8] = {
    1.000000f, 0.923880f, 0.707107f, 0.382683f,
    0.000000f, -0.382683f, -0.707107f, -0.923880f
};

__constant__ float TWIDDLE_16_SIN[8] = {
    0.000000f, -0.382683f, -0.707107f, -0.923880f,
    -1.000000f, -0.923880f, -0.707107f, -0.382683f
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
 * FFT16 kernel - 64 butterflies per block
 * Block: [64, 16] = 1024 threads
 */
__global__ void fft16_batch64_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int x = threadIdx.x;  // 0-63: which FFT butterfly
    const int y = threadIdx.y;  // 0-15: which point
    const int global_fft_id = blockIdx.x * 64 + x;
    
    if (global_fft_id >= num_windows) return;
    
    // Shared memory: [64 FFT][16 points] + padding
    __shared__ float2 shmem[64][18];
    
    // === LOAD with bit-reversal ===
    const int input_idx = global_fft_id * 16 + y;
    const int reversed_y = bitReverse4(y);
    shmem[x][reversed_y] = make_float2(input[input_idx].x, input[input_idx].y);
    __syncthreads();
    
    // ===================================================================
    // BUTTERFLY STAGES - LINEAR UNROLL
    // ===================================================================
    
    // STAGE 0: m=2, m2=1
    if (y < 8) {
        const int m2 = 1;
        const int k = (y / m2) * 2;
        const int j = y % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int twiddle_idx = j * 8;
        const float tw_cos = TWIDDLE_16_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_16_SIN[twiddle_idx];
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 1: m=4, m2=2
    if (y < 8) {
        const int m2 = 2;
        const int k = (y / m2) * 4;
        const int j = y % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int twiddle_idx = j * 4;
        const float tw_cos = TWIDDLE_16_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_16_SIN[twiddle_idx];
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 2: m=8, m2=4
    if (y < 8) {
        const int m2 = 4;
        const int k = (y / m2) * 8;
        const int j = y % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int twiddle_idx = j * 2;
        const float tw_cos = TWIDDLE_16_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_16_SIN[twiddle_idx];
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 3: m=16, m2=8 (final)
    if (y < 8) {
        const int m2 = 8;
        const int j = y;
        const int idx1 = j;
        const int idx2 = idx1 + m2;
        
        const int twiddle_idx = j;
        const float tw_cos = TWIDDLE_16_COS[twiddle_idx];
        const float tw_sin = TWIDDLE_16_SIN[twiddle_idx];
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // === STORE (NO shift!) ===
    const int output_idx = global_fft_id * 16 + y;
    output[output_idx] = make_cuComplex(shmem[x][y].x, shmem[x][y].y);
}

// Host launcher
extern "C" void launch_fft16_batch64(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(64, 16);  // [64 FFT, 16 points] = 1024 threads
    int num_blocks = (num_windows + 63) / 64;
    dim3 grid(num_blocks);
    
    fft16_batch64_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}



