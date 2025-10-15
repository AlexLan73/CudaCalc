/**
 * @file fft32_optimized_v3.cu
 * @brief FFT32 OPTIMIZED - minimal sync, register blocking
 * 
 * Optimizations:
 * - __ldg() for input
 * - Minimal __syncthreads() (only where absolutely needed)
 * - Register blocking where possible
 * - Padding +2
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

__constant__ float TWIDDLE_32_COS[16] = {
    1.000000f, 0.980785f, 0.923880f, 0.831470f,
    0.707107f, 0.555570f, 0.382683f, 0.195090f,
    0.000000f, -0.195090f, -0.382683f, -0.555570f,
   -0.707107f, -0.831470f, -0.923880f, -0.980785f
};

__constant__ float TWIDDLE_32_SIN[16] = {
    0.000000f, -0.195090f, -0.382683f, -0.555570f,
   -0.707107f, -0.831470f, -0.923880f, -0.980785f,
   -1.000000f, -0.980785f, -0.923880f, -0.831470f,
   -0.707107f, -0.555570f, -0.382683f, -0.195090f
};

static __device__ int bitReverse5_v3(int x) {
    int result = 0;
    result |= (x & 1) << 4;
    result |= (x & 2) << 2;
    result |= (x & 4);
    result |= (x & 8) >> 2;
    result |= (x & 16) >> 4;
    return result;
}

__global__ void fft32_optimized_v3_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int x = threadIdx.x;  // 0-31: which FFT
    const int y = threadIdx.y;  // 0-31: which point
    const int global_fft_id = blockIdx.x * 32 + x;
    
    if (global_fft_id >= num_windows) return;
    
    // Shared memory with padding
    __shared__ float2 shmem[32][34];
    
    // === LOAD with __ldg() and bit-reversal ===
    const int input_idx = global_fft_id * 32 + y;
    const int reversed_y = bitReverse5_v3(y);
    const cuComplex val = __ldg(&input[input_idx]);
    shmem[x][reversed_y] = make_float2(val.x, val.y);
    __syncthreads();  // SYNC 1: After load
    
    // ===================================================================
    // BUTTERFLY STAGES - Optimized for minimal sync
    // ===================================================================
    
    // STAGE 0: m=2, m2=1
    if (y < 16) {
        const int m2 = 1;
        const int k = (y / m2) * 2;
        const int j = y % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int tw_idx = j * 16;
        const float tw_cos = TWIDDLE_32_COS[tw_idx & 15];
        const float tw_sin = TWIDDLE_32_SIN[tw_idx & 15];
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();  // SYNC 2
    
    // STAGE 1: m=4, m2=2
    if (y < 16) {
        const int m2 = 2;
        const int k = (y / m2) * 4;
        const int j = y % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int tw_idx = j * 8;
        const float tw_cos = TWIDDLE_32_COS[tw_idx];
        const float tw_sin = TWIDDLE_32_SIN[tw_idx];
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();  // SYNC 3
    
    // STAGE 2: m=8, m2=4
    if (y < 16) {
        const int m2 = 4;
        const int k = (y / m2) * 8;
        const int j = y % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int tw_idx = j * 4;
        const float tw_cos = TWIDDLE_32_COS[tw_idx];
        const float tw_sin = TWIDDLE_32_SIN[tw_idx];
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();  // SYNC 4
    
    // STAGE 3: m=16, m2=8
    if (y < 16) {
        const int m2 = 8;
        const int k = (y / m2) * 16;
        const int j = y % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int tw_idx = j * 2;
        const float tw_cos = TWIDDLE_32_COS[tw_idx];
        const float tw_sin = TWIDDLE_32_SIN[tw_idx];
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();  // SYNC 5
    
    // STAGE 4: m=32, m2=16 (final)
    if (y < 16) {
        const int m2 = 16;
        const int j = y;
        const int idx1 = j;
        const int idx2 = idx1 + m2;
        
        const int tw_idx = j;
        const float tw_cos = TWIDDLE_32_COS[tw_idx];
        const float tw_sin = TWIDDLE_32_SIN[tw_idx];
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();  // SYNC 6 - before store
    
    // Store
    const int output_idx = global_fft_id * 32 + y;
    output[output_idx] = make_cuComplex(shmem[x][y].x, shmem[x][y].y);
}

extern "C" void launch_fft32_optimized_v3(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(32, 32);
    int num_blocks = (num_windows + 31) / 32;
    dim3 grid(num_blocks);
    
    fft32_optimized_v3_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}



