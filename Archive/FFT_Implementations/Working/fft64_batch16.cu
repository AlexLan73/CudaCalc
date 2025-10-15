#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// FFT64 twiddle factors (32 values for 6 stages)
__constant__ float cos_table_64[32];
__constant__ float sin_table_64[32];

// Initialize twiddle factors for FFT64
extern "C" void init_fft64_twiddles() {
    float h_cos[32], h_sin[32];
    
    // Stage 1: W_64^0, W_64^1, ..., W_64^31
    for (int i = 0; i < 32; ++i) {
        float angle = -2.0f * M_PI * i / 64.0f;
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbol(cos_table_64, h_cos, 32 * sizeof(float));
    cudaMemcpyToSymbol(sin_table_64, h_sin, 32 * sizeof(float));
}

// 6-bit bit reversal
__device__ __forceinline__ int bitReverse6(int x) {
    x = ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = ((x & 0xFFFF0000) >> 16) | ((x & 0x0000FFFF) << 16);
    return x >> 26; // Keep only 6 bits
}

__global__ void fft64_batch16_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: [16, 64] = 1024 threads
    // threadIdx.x = 0-15 (which FFT window)
    // threadIdx.y = 0-63 (which point in FFT64)
    
    const int window_id = threadIdx.x;
    const int point_id = threadIdx.y;
    const int block_id = blockIdx.x;
    
    const int global_window = block_id * 16 + window_id;
    if (global_window >= num_windows) return;
    
    // Shared memory: [16 windows][64 points] = 1024 complex numbers
    extern __shared__ cuComplex shmem[];
    cuComplex (*window_data)[64] = (cuComplex (*)[64])shmem;
    
    // Load input data with bit reversal
    const int input_idx = global_window * 64 + point_id;
    const int bit_rev_idx = bitReverse6(point_id);
    window_data[window_id][bit_rev_idx] = input[input_idx];
    
    __syncthreads();
    
    // FFT64: 6 stages of butterflies
    // Stage 1: 32 butterflies, distance 1
    if (point_id < 32) {
        const int pair = point_id + 32;
        cuComplex a = window_data[window_id][point_id];
        cuComplex b = window_data[window_id][pair];
        
        const float tw_real = cos_table_64[point_id];
        const float tw_imag = sin_table_64[point_id];
        
        const float b_tw_r = b.x * tw_real - b.y * tw_imag;
        const float b_tw_i = b.x * tw_imag + b.y * tw_real;
        
        window_data[window_id][point_id] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        window_data[window_id][pair] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Stage 2: 16 butterflies, distance 2
    if (point_id < 16) {
        const int pair = point_id + 16;
        cuComplex a = window_data[window_id][point_id];
        cuComplex b = window_data[window_id][pair];
        
        const int tw_idx = point_id * 2;
        const float tw_real = cos_table_64[tw_idx];
        const float tw_imag = sin_table_64[tw_idx];
        
        const float b_tw_r = b.x * tw_real - b.y * tw_imag;
        const float b_tw_i = b.x * tw_imag + b.y * tw_real;
        
        window_data[window_id][point_id] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        window_data[window_id][pair] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Stage 3: 8 butterflies, distance 4
    if (point_id < 8) {
        const int pair = point_id + 8;
        cuComplex a = window_data[window_id][point_id];
        cuComplex b = window_data[window_id][pair];
        
        const int tw_idx = point_id * 4;
        const float tw_real = cos_table_64[tw_idx];
        const float tw_imag = sin_table_64[tw_idx];
        
        const float b_tw_r = b.x * tw_real - b.y * tw_imag;
        const float b_tw_i = b.x * tw_imag + b.y * tw_real;
        
        window_data[window_id][point_id] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        window_data[window_id][pair] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Stage 4: 4 butterflies, distance 8
    if (point_id < 4) {
        const int pair = point_id + 4;
        cuComplex a = window_data[window_id][point_id];
        cuComplex b = window_data[window_id][pair];
        
        const int tw_idx = point_id * 8;
        const float tw_real = cos_table_64[tw_idx];
        const float tw_imag = sin_table_64[tw_idx];
        
        const float b_tw_r = b.x * tw_real - b.y * tw_imag;
        const float b_tw_i = b.x * tw_imag + b.y * tw_real;
        
        window_data[window_id][point_id] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        window_data[window_id][pair] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Stage 5: 2 butterflies, distance 16
    if (point_id < 2) {
        const int pair = point_id + 2;
        cuComplex a = window_data[window_id][point_id];
        cuComplex b = window_data[window_id][pair];
        
        const int tw_idx = point_id * 16;
        const float tw_real = cos_table_64[tw_idx];
        const float tw_imag = sin_table_64[tw_idx];
        
        const float b_tw_r = b.x * tw_real - b.y * tw_imag;
        const float b_tw_i = b.x * tw_imag + b.y * tw_real;
        
        window_data[window_id][point_id] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        window_data[window_id][pair] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Stage 6: 1 butterfly, distance 32
    if (point_id == 0) {
        const int pair = 1;
        cuComplex a = window_data[window_id][point_id];
        cuComplex b = window_data[window_id][pair];
        
        const int tw_idx = 0; // W_64^0 = 1
        const float tw_real = cos_table_64[tw_idx];
        const float tw_imag = sin_table_64[tw_idx];
        
        const float b_tw_r = b.x * tw_real - b.y * tw_imag;
        const float b_tw_i = b.x * tw_imag + b.y * tw_real;
        
        window_data[window_id][point_id] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        window_data[window_id][pair] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Store results
    const int output_idx = global_window * 64 + point_id;
    output[output_idx] = window_data[window_id][point_id];
}

extern "C" void launch_fft64_batch16(const cuComplex* input, cuComplex* output, int num_windows) {
    // Initialize twiddle factors
    static bool initialized = false;
    if (!initialized) {
        init_fft64_twiddles();
        initialized = true;
    }
    
    // Launch kernel: [16, 64] threads per block
    dim3 blockDim(16, 64);
    int num_blocks = (num_windows + 15) / 16; // 16 windows per block
    
    // Shared memory: 16 windows * 64 points * sizeof(cuComplex)
    size_t shared_mem_size = 16 * 64 * sizeof(cuComplex);
    
    fft64_batch16_kernel<<<num_blocks, blockDim, shared_mem_size>>>(
        input, output, num_windows
    );
}
