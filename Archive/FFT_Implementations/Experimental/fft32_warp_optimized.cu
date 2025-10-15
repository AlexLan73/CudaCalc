#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Pre-computed twiddle factors for FFT32
__constant__ cuComplex fft32_twiddles[16] = {
    {1.000000f, 0.000000f}, {0.980785f, -0.195090f}, {0.923880f, -0.382683f}, {0.831470f, -0.555570f},
    {0.707107f, -0.707107f}, {0.555570f, -0.831470f}, {0.382683f, -0.923880f}, {0.195090f, -0.980785f},
    {0.000000f, -1.000000f}, {-0.195090f, -0.980785f}, {-0.382683f, -0.923880f}, {-0.555570f, -0.831470f},
    {-0.707107f, -0.707107f}, {-0.831470f, -0.555570f}, {-0.923880f, -0.382683f}, {-0.980785f, -0.195090f}
};

// Warp-level butterfly operation using shuffle
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

// Warp-level bit reversal using shuffle
__device__ __forceinline__ int warp_bit_reverse_5(int x) {
    // Reverse 5 bits: 0bABCDE -> 0bEDCBA
    x = __brev(x) >> 27;  // Reverse all 32 bits, then shift right by 27
    return x;
}

// Warp-level FFT32 kernel
__global__ void fft32_warp_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    int global_window = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_window >= num_windows) return;
    
    int local_thread = threadIdx.x;
    int warp_id = local_thread / 32;
    int lane_id = local_thread % 32;
    
    // Load input data with bit reversal using warp shuffle
    cuComplex data[2];  // Each thread handles 2 points for FFT32
    
    if (lane_id < 16) {
        // Load first 16 points
        int bit_rev_idx = warp_bit_reverse_5(lane_id);
        data[0] = input[global_window * 32 + bit_rev_idx];
        
        // Load second 16 points
        bit_rev_idx = warp_bit_reverse_5(lane_id + 16);
        data[1] = input[global_window * 32 + bit_rev_idx];
    }
    
    // FFT32 has 5 stages, each stage processes different pairs
    // Stage 1: distance 16
    if (lane_id < 16) {
        int pair_lane = lane_id + 16;
        cuComplex twiddle = fft32_twiddles[0];  // W_32^0 = 1
        
        // Get pair data using shuffle
        float pair_x = __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane);
        float pair_y = __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane);
        cuComplex pair = {pair_x, pair_y};
        
        warp_butterfly(data[0], pair, twiddle);
        
        // Store back using shuffle
        __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane, pair.x);
        __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane, pair.y);
    }
    
    // Stage 2: distance 8
    if (lane_id < 16) {
        int pair_lane = lane_id + 8;
        cuComplex twiddle = fft32_twiddles[lane_id % 2 * 8];  // W_32^(lane_id*2)
        
        float pair_x = __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane);
        float pair_y = __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane);
        cuComplex pair = {pair_x, pair_y};
        
        warp_butterfly(data[0], pair, twiddle);
        
        __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane, pair.x);
        __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane, pair.y);
    }
    
    // Stage 3: distance 4
    if (lane_id < 16) {
        int pair_lane = lane_id + 4;
        cuComplex twiddle = fft32_twiddles[lane_id % 4 * 4];  // W_32^(lane_id*4)
        
        float pair_x = __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane);
        float pair_y = __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane);
        cuComplex pair = {pair_x, pair_y};
        
        warp_butterfly(data[0], pair, twiddle);
        
        __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane, pair.x);
        __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane, pair.y);
    }
    
    // Stage 4: distance 2
    if (lane_id < 16) {
        int pair_lane = lane_id + 2;
        cuComplex twiddle = fft32_twiddles[lane_id % 8 * 2];  // W_32^(lane_id*8)
        
        float pair_x = __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane);
        float pair_y = __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane);
        cuComplex pair = {pair_x, pair_y};
        
        warp_butterfly(data[0], pair, twiddle);
        
        __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane, pair.x);
        __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane, pair.y);
    }
    
    // Stage 5: distance 1
    if (lane_id < 16) {
        int pair_lane = lane_id + 1;
        cuComplex twiddle = fft32_twiddles[lane_id % 16];  // W_32^(lane_id*16)
        
        float pair_x = __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane);
        float pair_y = __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane);
        cuComplex pair = {pair_x, pair_y};
        
        warp_butterfly(data[0], pair, twiddle);
        
        __shfl_sync(0xFFFFFFFF, data[0].x, pair_lane, pair.x);
        __shfl_sync(0xFFFFFFFF, data[0].y, pair_lane, pair.y);
    }
    
    // Store results
    if (lane_id < 16) {
        output[global_window * 32 + lane_id] = data[0];
        output[global_window * 32 + lane_id + 16] = data[1];
    }
}

extern "C" void launch_fft32_warp(const cuComplex* input, cuComplex* output, int num_windows) {
    dim3 blockDim(32);  // One warp per block
    dim3 gridDim((num_windows + blockDim.x - 1) / blockDim.x);
    
    fft32_warp_kernel<<<gridDim, blockDim>>>(input, output, num_windows);
}


