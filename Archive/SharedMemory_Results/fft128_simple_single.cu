/**
 * @file fft128_simple_single.cu
 * @brief FFT128 Simple version - 1 FFT per block
 * 
 * Честное сравнение с Shared Memory версией
 * 1 FFT128 на блок, 128 потоков на блок
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// FFT128 twiddle factors (64 values)
__constant__ float cos_table_128_single[64];
__constant__ float sin_table_128_single[64];

// Initialize twiddle factors for FFT128
void init_fft128_twiddles_single() {
    float h_cos[64], h_sin[64];
    
    for (int i = 0; i < 64; ++i) {
        float angle = -2.0f * M_PI * i / 128.0f;
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbol(cos_table_128_single, h_cos, 64 * sizeof(float));
    cudaMemcpyToSymbol(sin_table_128_single, h_sin, 64 * sizeof(float));
}

// 7-bit bit reversal
__device__ __forceinline__ int bitReverse7_single(int x) {
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__global__ void fft128_simple_single_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: 128 threads per block, 1 FFT per block
    const int fft_id = blockIdx.x;
    const int tid = threadIdx.x;  // 0-127
    
    if (fft_id >= num_windows) return;
    
    // Shared memory: 128 points (same as Shared Memory version)
    __shared__ cuComplex data[128];
    
    // Load input data with bit reversal
    const int input_idx = fft_id * 128 + tid;
    const int bit_rev_idx = bitReverse7_single(tid);
    data[bit_rev_idx] = input[input_idx];
    
    __syncthreads();
    
    // FFT128: 7 stages (точно такой же алгоритм как в Shared Memory версии)
    // Stage 1: m=2, m2=1, distance=1, 64 butterflies
    {
        const int m = 2;
        const int m2 = 1;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128_single[tw_idx];
            const float tw_imag = sin_table_128_single[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 2: m=4, m2=2, distance=2, 64 butterflies
    {
        const int m = 4;
        const int m2 = 2;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128_single[tw_idx];
            const float tw_imag = sin_table_128_single[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 3: m=8, m2=4, distance=4, 64 butterflies
    {
        const int m = 8;
        const int m2 = 4;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128_single[tw_idx];
            const float tw_imag = sin_table_128_single[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 4: m=16, m2=8, distance=8, 64 butterflies
    {
        const int m = 16;
        const int m2 = 8;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128_single[tw_idx];
            const float tw_imag = sin_table_128_single[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 5: m=32, m2=16, distance=16, 64 butterflies
    {
        const int m = 32;
        const int m2 = 16;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128_single[tw_idx];
            const float tw_imag = sin_table_128_single[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 6: m=64, m2=32, distance=32, 64 butterflies
    {
        const int m = 64;
        const int m2 = 32;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128_single[tw_idx];
            const float tw_imag = sin_table_128_single[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 7: m=128, m2=64, distance=64, 64 butterflies
    {
        const int m = 128;
        const int m2 = 64;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128_single[tw_idx];
            const float tw_imag = sin_table_128_single[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Store result
    const int output_idx = fft_id * 128 + tid;
    output[output_idx] = data[tid];
}

// Host launcher
extern "C" void launch_fft128_simple_single(const cuComplex* input, cuComplex* output, int num_windows) {
    // Initialize twiddle factors
    init_fft128_twiddles_single();
    
    dim3 block(128);  // 128 threads per block, 1 FFT per block
    dim3 grid(num_windows);  // One block per FFT
    
    fft128_simple_single_kernel<<<grid, block>>>(input, output, num_windows);
}
