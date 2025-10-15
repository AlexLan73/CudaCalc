#include <cuda_runtime.h>
#include <cuComplex.h>

// Pre-computed twiddle factors for FFT32
__constant__ cuComplex fft32_twiddles[16] = {
    {1.000000f, 0.000000f}, {0.980785f, -0.195090f}, {0.923880f, -0.382683f}, {0.831470f, -0.555570f},
    {0.707107f, -0.707107f}, {0.555570f, -0.831470f}, {0.382683f, -0.923880f}, {0.195090f, -0.980785f},
    {0.000000f, -1.000000f}, {-0.195090f, -0.980785f}, {-0.382683f, -0.923880f}, {-0.555570f, -0.831470f},
    {-0.707107f, -0.707107f}, {-0.831470f, -0.555570f}, {-0.923880f, -0.382683f}, {-0.980785f, -0.195090f}
};

// Warp-level butterfly operation
__device__ __forceinline__ void warp_butterfly(cuComplex& a, cuComplex& b, cuComplex twiddle) {
    // Complex multiplication: b * twiddle
    float b_tw_r = b.x * twiddle.x - b.y * twiddle.y;
    float b_tw_i = b.x * twiddle.y + b.y * twiddle.x;
    
    // Butterfly: a' = a + b*tw, b' = a - b*tw
    cuComplex temp_a = a;
    a.x = temp_a.x + b_tw_r;
    a.y = temp_a.y + b_tw_i;
    b.x = temp_a.x - b_tw_r;
    b.y = temp_a.y - b_tw_i;
}

// Bit reversal for 5 bits
__device__ __forceinline__ int bit_reverse_5(int x) {
    x = ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = ((x & 0xFFFF0000) >> 16) | ((x & 0x0000FFFF) << 16);
    return x >> 27;  // Keep only 5 bits
}

// Simplified warp-level FFT32 kernel
__global__ void fft32_warp_v2_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    int global_window = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_window >= num_windows) return;
    
    int lane_id = threadIdx.x % 32;
    
    // Each thread handles one point, use shared memory for communication
    __shared__ cuComplex shmem[32];
    
    // Load input with bit reversal
    int bit_rev_idx = bit_reverse_5(lane_id);
    shmem[lane_id] = input[global_window * 32 + bit_rev_idx];
    __syncthreads();
    
    // FFT32 stages using warp shuffle
    cuComplex my_data = shmem[lane_id];
    
    // Stage 1: distance 16
    if (lane_id < 16) {
        cuComplex pair_data;
        pair_data.x = __shfl_sync(0xFFFFFFFF, my_data.x, lane_id + 16);
        pair_data.y = __shfl_sync(0xFFFFFFFF, my_data.y, lane_id + 16);
        
        warp_butterfly(my_data, pair_data, fft32_twiddles[0]);
        
        // Store back
        shmem[lane_id] = my_data;
        shmem[lane_id + 16] = pair_data;
    }
    __syncthreads();
    
    // Stage 2: distance 8
    if (lane_id < 16) {
        my_data = shmem[lane_id];
        cuComplex pair_data;
        pair_data.x = __shfl_sync(0xFFFFFFFF, my_data.x, lane_id + 8);
        pair_data.y = __shfl_sync(0xFFFFFFFF, my_data.y, lane_id + 8);
        
        int twiddle_idx = (lane_id % 2) * 8;
        warp_butterfly(my_data, pair_data, fft32_twiddles[twiddle_idx]);
        
        shmem[lane_id] = my_data;
        shmem[lane_id + 8] = pair_data;
    }
    __syncthreads();
    
    // Stage 3: distance 4
    if (lane_id < 16) {
        my_data = shmem[lane_id];
        cuComplex pair_data;
        pair_data.x = __shfl_sync(0xFFFFFFFF, my_data.x, lane_id + 4);
        pair_data.y = __shfl_sync(0xFFFFFFFF, my_data.y, lane_id + 4);
        
        int twiddle_idx = (lane_id % 4) * 4;
        warp_butterfly(my_data, pair_data, fft32_twiddles[twiddle_idx]);
        
        shmem[lane_id] = my_data;
        shmem[lane_id + 4] = pair_data;
    }
    __syncthreads();
    
    // Stage 4: distance 2
    if (lane_id < 16) {
        my_data = shmem[lane_id];
        cuComplex pair_data;
        pair_data.x = __shfl_sync(0xFFFFFFFF, my_data.x, lane_id + 2);
        pair_data.y = __shfl_sync(0xFFFFFFFF, my_data.y, lane_id + 2);
        
        int twiddle_idx = (lane_id % 8) * 2;
        warp_butterfly(my_data, pair_data, fft32_twiddles[twiddle_idx]);
        
        shmem[lane_id] = my_data;
        shmem[lane_id + 2] = pair_data;
    }
    __syncthreads();
    
    // Stage 5: distance 1
    if (lane_id < 16) {
        my_data = shmem[lane_id];
        cuComplex pair_data;
        pair_data.x = __shfl_sync(0xFFFFFFFF, my_data.x, lane_id + 1);
        pair_data.y = __shfl_sync(0xFFFFFFFF, my_data.y, lane_id + 1);
        
        int twiddle_idx = lane_id % 16;
        warp_butterfly(my_data, pair_data, fft32_twiddles[twiddle_idx]);
        
        shmem[lane_id] = my_data;
        shmem[lane_id + 1] = pair_data;
    }
    __syncthreads();
    
    // Store results
    output[global_window * 32 + lane_id] = shmem[lane_id];
}

extern "C" void launch_fft32_warp_v2(const cuComplex* input, cuComplex* output, int num_windows) {
    dim3 blockDim(32);  // One warp per block
    dim3 gridDim((num_windows + blockDim.x - 1) / blockDim.x);
    
    fft32_warp_v2_kernel<<<gridDim, blockDim>>>(input, output, num_windows);
}


