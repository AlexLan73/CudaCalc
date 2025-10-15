/**
 * @file fft256_shared_memory.cu
 * @brief FFT256 GPU kernel using shared memory - 1 FFT per block
 * 
 * Точная копия рабочего fft256_simple_correct.cu алгоритма,
 * адаптированная для shared memory: 1 FFT256 на блок
 * Shared memory: 256 точек × 8 байт = 2048 байт
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// FFT256 twiddle factors (128 values) - копируем из рабочего алгоритма
__constant__ float cos_table_256_shared[128];
__constant__ float sin_table_256_shared[128];

// Initialize twiddle factors for FFT256
void init_fft256_twiddles_shared() {
    float h_cos[128], h_sin[128];
    
    for (int i = 0; i < 128; ++i) {
        float angle = -2.0f * M_PI * i / 256.0f;
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbol(cos_table_256_shared, h_cos, 128 * sizeof(float));
    cudaMemcpyToSymbol(sin_table_256_shared, h_sin, 128 * sizeof(float));
}

// 8-bit bit reversal (копируем из рабочего алгоритма)
__device__ __forceinline__ int bitReverse8_shared(int x) {
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__global__ void fft256_shared_memory_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: 256 threads per block, 1 FFT per block
    const int fft_id = blockIdx.x;
    const int tid = threadIdx.x;  // 0-255
    
    if (fft_id >= num_windows) return;
    
    // Shared memory: 256 points
    __shared__ cuComplex data[256];
    
    // Load input data with bit reversal
    const int input_idx = fft_id * 256 + tid;
    const int bit_rev_idx = bitReverse8_shared(tid);
    data[bit_rev_idx] = input[input_idx];
    
    __syncthreads();
    
    // FFT256: 8 stages (копируем точно из рабочего алгоритма)
    // Stage 1: m=2, m2=1, distance=1, 128 butterflies
    {
        const int m = 2;
        const int m2 = 1;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (256 / m);
            const float tw_real = cos_table_256_shared[tw_idx];
            const float tw_imag = sin_table_256_shared[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 2: m=4, m2=2, distance=2, 128 butterflies
    {
        const int m = 4;
        const int m2 = 2;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (256 / m);
            const float tw_real = cos_table_256_shared[tw_idx];
            const float tw_imag = sin_table_256_shared[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 3: m=8, m2=4, distance=4, 128 butterflies
    {
        const int m = 8;
        const int m2 = 4;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (256 / m);
            const float tw_real = cos_table_256_shared[tw_idx];
            const float tw_imag = sin_table_256_shared[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 4: m=16, m2=8, distance=8, 128 butterflies
    {
        const int m = 16;
        const int m2 = 8;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (256 / m);
            const float tw_real = cos_table_256_shared[tw_idx];
            const float tw_imag = sin_table_256_shared[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 5: m=32, m2=16, distance=16, 128 butterflies
    {
        const int m = 32;
        const int m2 = 16;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (256 / m);
            const float tw_real = cos_table_256_shared[tw_idx];
            const float tw_imag = sin_table_256_shared[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 6: m=64, m2=32, distance=32, 128 butterflies
    {
        const int m = 64;
        const int m2 = 32;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (256 / m);
            const float tw_real = cos_table_256_shared[tw_idx];
            const float tw_imag = sin_table_256_shared[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 7: m=128, m2=64, distance=64, 128 butterflies
    {
        const int m = 128;
        const int m2 = 64;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (256 / m);
            const float tw_real = cos_table_256_shared[tw_idx];
            const float tw_imag = sin_table_256_shared[tw_idx];
            
            cuComplex a = data[idx1];
            cuComplex b = data[idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            data[idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            data[idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 8: m=256, m2=128, distance=128, 128 butterflies
    {
        const int m = 256;
        const int m2 = 128;
        const int group = tid / m;
        const int idx_in_group = tid % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (256 / m);
            const float tw_real = cos_table_256_shared[tw_idx];
            const float tw_imag = sin_table_256_shared[tw_idx];
            
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
    const int output_idx = fft_id * 256 + tid;
    output[output_idx] = data[tid];
}

// Host launcher
extern "C" void launch_fft256_shared_memory(const cuComplex* input, cuComplex* output, int num_windows) {
    // Initialize twiddle factors
    init_fft256_twiddles_shared();
    
    dim3 block(256);  // 256 threads per block, 1 FFT per block
    dim3 grid(num_windows);  // One block per FFT
    
    fft256_shared_memory_kernel<<<grid, block>>>(input, output, num_windows);
}


