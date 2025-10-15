#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// FFT128 twiddle factors (64 values)
__constant__ float cos_table_128[64];
__constant__ float sin_table_128[64];

// Initialize twiddle factors for FFT128
void init_fft128_twiddles() {
    float h_cos[64], h_sin[64];
    
    for (int i = 0; i < 64; ++i) {
        float angle = -2.0f * M_PI * i / 128.0f;
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbol(cos_table_128, h_cos, 64 * sizeof(float));
    cudaMemcpyToSymbol(sin_table_128, h_sin, 64 * sizeof(float));
}

// 7-bit bit reversal
__device__ __forceinline__ int bitReverse7(int x) {
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__global__ void fft128_simple_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: [8, 128] = 1024 threads
    // threadIdx.x = 0-7 (which FFT window)
    // threadIdx.y = 0-127 (which point in FFT128)
    
    const int window_id = threadIdx.x;
    const int point_id = threadIdx.y;
    const int block_id = blockIdx.x;
    
    const int global_window = block_id * 8 + window_id;
    if (global_window >= num_windows) return;
    
    // Shared memory: [8 windows][128 points]
    extern __shared__ cuComplex shmem[];
    cuComplex (*window_data)[128] = (cuComplex (*)[128])shmem;
    
    // Load input data with bit reversal
    const int input_idx = global_window * 128 + point_id;
    const int bit_rev_idx = bitReverse7(point_id);
    window_data[window_id][bit_rev_idx] = input[input_idx];
    
    __syncthreads();
    
    // FFT128: 7 stages
    // Stage 1: m=2, m2=1, distance=1, 64 butterflies
    {
        const int m = 2;
        const int m2 = 1;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128[tw_idx];
            const float tw_imag = sin_table_128[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 2: m=4, m2=2, distance=2, 64 butterflies
    {
        const int m = 4;
        const int m2 = 2;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128[tw_idx];
            const float tw_imag = sin_table_128[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 3: m=8, m2=4, distance=4, 64 butterflies
    {
        const int m = 8;
        const int m2 = 4;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128[tw_idx];
            const float tw_imag = sin_table_128[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 4: m=16, m2=8, distance=8, 64 butterflies
    {
        const int m = 16;
        const int m2 = 8;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128[tw_idx];
            const float tw_imag = sin_table_128[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 5: m=32, m2=16, distance=16, 64 butterflies
    {
        const int m = 32;
        const int m2 = 16;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128[tw_idx];
            const float tw_imag = sin_table_128[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 6: m=64, m2=32, distance=32, 64 butterflies
    {
        const int m = 64;
        const int m2 = 32;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (128 / m);
            const float tw_real = cos_table_128[tw_idx];
            const float tw_imag = sin_table_128[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 7: m=128, m2=64, distance=64, 64 butterflies
    if (point_id < 64) {
        const int idx1 = point_id;
        const int idx2 = point_id + 64;
        
        const int tw_idx = point_id;
        const float tw_real = cos_table_128[tw_idx];
        const float tw_imag = sin_table_128[tw_idx];
        
        cuComplex a = window_data[window_id][idx1];
        cuComplex b = window_data[window_id][idx2];
        
        const float b_tw_r = b.x * tw_real - b.y * tw_imag;
        const float b_tw_i = b.x * tw_imag + b.y * tw_real;
        
        window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Store results
    const int output_idx = global_window * 128 + point_id;
    output[output_idx] = window_data[window_id][point_id];
}

extern "C" void launch_fft128_simple(const cuComplex* input, cuComplex* output, int num_windows) {
    static bool initialized = false;
    if (!initialized) {
        init_fft128_twiddles();
        initialized = true;
    }
    
    dim3 blockDim(8, 128);
    int num_blocks = (num_windows + 7) / 8;
    size_t shared_mem_size = 8 * 128 * sizeof(cuComplex);
    
    fft128_simple_kernel<<<num_blocks, blockDim, shared_mem_size>>>(
        input, output, num_windows
    );
}


