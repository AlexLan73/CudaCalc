/**
 * @file fft128_shared_memory.cu
 * @brief FFT128 GPU kernel using shared memory - 1 FFT per block
 * 
 * Точная копия рабочего fft128_simple_correct.cu алгоритма,
 * адаптированная для shared memory: 1 FFT128 на блок
 * Shared memory: 128 точек × 8 байт = 1024 байт
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// FFT128 twiddle factors (64 values) - копируем из рабочего алгоритма
__constant__ float cos_table_128_shared[64];
__constant__ float sin_table_128_shared[64];

// Initialize twiddle factors for FFT128
void init_fft128_twiddles_shared() {
    float h_cos[64], h_sin[64];
    
    for (int i = 0; i < 64; ++i) {
        float angle = -2.0f * M_PI * i / 128.0f;
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbol(cos_table_128_shared, h_cos, 64 * sizeof(float));
    cudaMemcpyToSymbol(sin_table_128_shared, h_sin, 64 * sizeof(float));
}

// 7-bit bit reversal (копируем из рабочего алгоритма)
__device__ __forceinline__ int bitReverse7_shared(int x) {
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__global__ void fft128_shared_memory_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: 128 threads per block, 1 FFT per block
    const int fft_id = blockIdx.x;
    const int tid = threadIdx.x;  // 0-127
    
    if (fft_id >= num_windows) return;
    
    // Shared memory: 128 points
    __shared__ cuComplex data[128];
    
    // Load input data with bit reversal
    const int input_idx = fft_id * 128 + tid;
    const int bit_rev_idx = bitReverse7_shared(tid);
    data[bit_rev_idx] = input[input_idx];
    
    __syncthreads();
    
    // FFT128: 7 stages (копируем точно из рабочего алгоритма)
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
            const float tw_real = cos_table_128_shared[tw_idx];
            const float tw_imag = sin_table_128_shared[tw_idx];
            
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
            const float tw_real = cos_table_128_shared[tw_idx];
            const float tw_imag = sin_table_128_shared[tw_idx];
            
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
            const float tw_real = cos_table_128_shared[tw_idx];
            const float tw_imag = sin_table_128_shared[tw_idx];
            
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
            const float tw_real = cos_table_128_shared[tw_idx];
            const float tw_imag = sin_table_128_shared[tw_idx];
            
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
            const float tw_real = cos_table_128_shared[tw_idx];
            const float tw_imag = sin_table_128_shared[tw_idx];
            
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
            const float tw_real = cos_table_128_shared[tw_idx];
            const float tw_imag = sin_table_128_shared[tw_idx];
            
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
            const float tw_real = cos_table_128_shared[tw_idx];
            const float tw_imag = sin_table_128_shared[tw_idx];
            
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
extern "C" void launch_fft128_shared_memory(const cuComplex* input, cuComplex* output, int num_windows) {
    // Initialize twiddle factors
    init_fft128_twiddles_shared();
    
    dim3 block(128);  // 128 threads per block, 1 FFT per block
    dim3 grid(num_windows);  // One block per FFT
    
    fft128_shared_memory_kernel<<<grid, block>>>(input, output, num_windows);
}


