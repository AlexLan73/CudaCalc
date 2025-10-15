/**
 * @file fft64_shared_memory.cu
 * @brief FFT64 GPU kernel using shared memory
 * 
 * Точная копия рабочего fft64_simple_correct.cu алгоритма,
 * адаптированная для shared memory
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// FFT64 twiddle factors (32 values) - копируем из рабочего алгоритма
__constant__ float cos_table_64[32];
__constant__ float sin_table_64[32];

// Initialize twiddle factors for FFT64 (shared memory version)
void init_fft64_twiddles_shared() {
    float h_cos[32], h_sin[32];
    
    for (int i = 0; i < 32; ++i) {
        float angle = -2.0f * M_PI * i / 64.0f;
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbol(cos_table_64, h_cos, 32 * sizeof(float));
    cudaMemcpyToSymbol(sin_table_64, h_sin, 32 * sizeof(float));
}

// 6-bit bit reversal (копируем из рабочего алгоритма)
__device__ __forceinline__ int bitReverse6_shared(int x) {
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__global__ void fft64_shared_memory_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: [16, 64] = 1024 threads
    // threadIdx.x = 0-15 (which FFT window)
    // threadIdx.y = 0-63 (which point in FFT64)
    
    const int window_id = threadIdx.x;
    const int point_id = threadIdx.y;
    const int block_id = blockIdx.x;
    
    const int global_window = block_id * 16 + window_id;
    if (global_window >= num_windows) return;
    
    // Shared memory: [16 windows][64 points]
    extern __shared__ cuComplex shmem[];
    cuComplex (*window_data)[64] = (cuComplex (*)[64])shmem;
    
    // Load input data with bit reversal
    const int input_idx = global_window * 64 + point_id;
    const int bit_rev_idx = bitReverse6_shared(point_id);
    window_data[window_id][bit_rev_idx] = input[input_idx];
    
    __syncthreads();
    
    // FFT64: 6 stages (копируем точно из рабочего алгоритма)
    // Stage 1: m=2, m2=1, distance=1, 32 butterflies
    {
        const int m = 2;
        const int m2 = 1;
        const int group = point_id / m;  // which group (0-31)
        const int idx_in_group = point_id % m;  // position in group (0-1)
        
        if (idx_in_group < m2) {  // only first half of each group does work
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (64 / m);  // twiddle index
            const float tw_real = cos_table_64[tw_idx];
            const float tw_imag = sin_table_64[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 2: m=4, m2=2, distance=2, 32 butterflies
    {
        const int m = 4;
        const int m2 = 2;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (64 / m);
            const float tw_real = cos_table_64[tw_idx];
            const float tw_imag = sin_table_64[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 3: m=8, m2=4, distance=4, 32 butterflies
    {
        const int m = 8;
        const int m2 = 4;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (64 / m);
            const float tw_real = cos_table_64[tw_idx];
            const float tw_imag = sin_table_64[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 4: m=16, m2=8, distance=8, 32 butterflies
    {
        const int m = 16;
        const int m2 = 8;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (64 / m);
            const float tw_real = cos_table_64[tw_idx];
            const float tw_imag = sin_table_64[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 5: m=32, m2=16, distance=16, 32 butterflies
    {
        const int m = 32;
        const int m2 = 16;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (64 / m);
            const float tw_real = cos_table_64[tw_idx];
            const float tw_imag = sin_table_64[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 6: m=64, m2=32, distance=32, 32 butterflies
    {
        const int m = 64;
        const int m2 = 32;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (64 / m);
            const float tw_real = cos_table_64[tw_idx];
            const float tw_imag = sin_table_64[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Store result
    const int output_idx = global_window * 64 + point_id;
    output[output_idx] = window_data[window_id][point_id];
}

// Host launcher
extern "C" void launch_fft64_shared_memory(const cuComplex* input, cuComplex* output, int num_windows) {
    // Initialize twiddle factors
    init_fft64_twiddles_shared();
    
    dim3 block(16, 64);  // 16 windows, 64 points per window
    dim3 grid((num_windows + 15) / 16);  // Number of blocks needed
    
    // Shared memory size: 16 windows × 64 points × 8 bytes = 8192 bytes
    size_t shared_mem_size = 16 * 64 * sizeof(cuComplex);
    
    fft64_shared_memory_kernel<<<grid, block, shared_mem_size>>>(input, output, num_windows);
}
