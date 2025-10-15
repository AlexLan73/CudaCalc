/**
 * @file fft32_memory_opt.cu
 * @brief FFT32 with different shared memory optimizations
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

__device__ int bitReverse5(int x) {
    int result = 0;
    result |= (x & 1) << 4;
    result |= (x & 2) << 2;
    result |= (x & 4);
    result |= (x & 8) >> 2;
    result |= (x & 16) >> 4;
    return result;
}

// Version 1: No padding (baseline for comparison)
__global__ void fft32_no_padding_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int global_fft_id = blockIdx.x * 32 + x;
    
    if (global_fft_id >= num_windows) return;
    
    __shared__ float2 shmem[32][32];  // NO padding
    
    const int input_idx = global_fft_id * 32 + y;
    const int reversed_y = bitReverse5(y);
    shmem[x][reversed_y] = make_float2(input[input_idx].x, input[input_idx].y);
    __syncthreads();
    
    // ... (same butterfly stages as v2)
    // For brevity, just store
    
    const int output_idx = global_fft_id * 32 + y;
    output[output_idx] = make_cuComplex(shmem[x][y].x, shmem[x][y].y);
}

// Version 2: Padding +2 (current best)
__global__ void fft32_padding2_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int global_fft_id = blockIdx.x * 32 + x;
    
    if (global_fft_id >= num_windows) return;
    
    __shared__ float2 shmem[32][34];  // +2 padding
    
    // Same as baseline v2...
    const int input_idx = global_fft_id * 32 + y;
    const int reversed_y = bitReverse5(y);
    shmem[x][reversed_y] = make_float2(input[input_idx].x, input[input_idx].y);
    __syncthreads();
    
    const int output_idx = global_fft_id * 32 + y;
    output[output_idx] = make_cuComplex(shmem[x][y].x, shmem[x][y].y);
}

// Version 3: Padding +4
__global__ void fft32_padding4_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int global_fft_id = blockIdx.x * 32 + x;
    
    if (global_fft_id >= num_windows) return;
    
    __shared__ float2 shmem[32][36];  // +4 padding
    
    const int input_idx = global_fft_id * 32 + y;
    const int reversed_y = bitReverse5(y);
    shmem[x][reversed_y] = make_float2(input[input_idx].x, input[input_idx].y);
    __syncthreads();
    
    const int output_idx = global_fft_id * 32 + y;
    output[output_idx] = make_cuComplex(shmem[x][y].x, shmem[x][y].y);
}



