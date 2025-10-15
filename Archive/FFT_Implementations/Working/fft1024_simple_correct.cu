#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// FFT1024 twiddle factors (512 values)
__constant__ float cos_table_1024[512];
__constant__ float sin_table_1024[512];

// Initialize twiddle factors for FFT1024
void init_fft1024_twiddles() {
    float h_cos[512], h_sin[512];
    
    for (int i = 0; i < 512; ++i) {
        float angle = -2.0f * M_PI * i / 1024.0f;
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbol(cos_table_1024, h_cos, 512 * sizeof(float));
    cudaMemcpyToSymbol(sin_table_1024, h_sin, 512 * sizeof(float));
}

// 10-bit bit reversal
__device__ __forceinline__ int bitReverse10(int x) {
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__global__ void fft1024_simple_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: [1, 1024] = 1024 threads
    // threadIdx.x = 0 (which FFT window)
    // threadIdx.y = 0-1023 (which point in FFT1024)
    
    const int window_id = threadIdx.x;
    const int point_id = threadIdx.y;
    const int block_id = blockIdx.x;
    
    const int global_window = block_id * 1 + window_id;
    if (global_window >= num_windows) return;
    
    // Shared memory: [1 window][1024 points]
    extern __shared__ cuComplex shmem[];
    cuComplex (*window_data)[1024] = (cuComplex (*)[1024])shmem;
    
    // Load input data with bit reversal
    const int input_idx = global_window * 1024 + point_id;
    const int bit_rev_idx = bitReverse10(point_id);
    window_data[window_id][bit_rev_idx] = input[input_idx];
    
    __syncthreads();
    
    // FFT1024: 10 stages
    // Stage 1: m=2, m2=1, distance=1, 512 butterflies
    {
        const int m = 2;
        const int m2 = 1;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (1024 / m);
            const float tw_real = cos_table_1024[tw_idx];
            const float tw_imag = sin_table_1024[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 2: m=4, m2=2, distance=2, 512 butterflies
    {
        const int m = 4;
        const int m2 = 2;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (1024 / m);
            const float tw_real = cos_table_1024[tw_idx];
            const float tw_imag = sin_table_1024[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 3: m=8, m2=4, distance=4, 512 butterflies
    {
        const int m = 8;
        const int m2 = 4;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (1024 / m);
            const float tw_real = cos_table_1024[tw_idx];
            const float tw_imag = sin_table_1024[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 4: m=16, m2=8, distance=8, 512 butterflies
    {
        const int m = 16;
        const int m2 = 8;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (1024 / m);
            const float tw_real = cos_table_1024[tw_idx];
            const float tw_imag = sin_table_1024[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 5: m=32, m2=16, distance=16, 512 butterflies
    {
        const int m = 32;
        const int m2 = 16;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (1024 / m);
            const float tw_real = cos_table_1024[tw_idx];
            const float tw_imag = sin_table_1024[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 6: m=64, m2=32, distance=32, 512 butterflies
    {
        const int m = 64;
        const int m2 = 32;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (1024 / m);
            const float tw_real = cos_table_1024[tw_idx];
            const float tw_imag = sin_table_1024[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 7: m=128, m2=64, distance=64, 512 butterflies
    {
        const int m = 128;
        const int m2 = 64;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (1024 / m);
            const float tw_real = cos_table_1024[tw_idx];
            const float tw_imag = sin_table_1024[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 8: m=256, m2=128, distance=128, 512 butterflies
    {
        const int m = 256;
        const int m2 = 128;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (1024 / m);
            const float tw_real = cos_table_1024[tw_idx];
            const float tw_imag = sin_table_1024[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 9: m=512, m2=256, distance=256, 512 butterflies
    {
        const int m = 512;
        const int m2 = 256;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (1024 / m);
            const float tw_real = cos_table_1024[tw_idx];
            const float tw_imag = sin_table_1024[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 10: m=1024, m2=512, distance=512, 512 butterflies
    if (point_id < 512) {
        const int idx1 = point_id;
        const int idx2 = point_id + 512;
        
        const int tw_idx = point_id;
        const float tw_real = cos_table_1024[tw_idx];
        const float tw_imag = sin_table_1024[tw_idx];
        
        cuComplex a = window_data[window_id][idx1];
        cuComplex b = window_data[window_id][idx2];
        
        const float b_tw_r = b.x * tw_real - b.y * tw_imag;
        const float b_tw_i = b.x * tw_imag + b.y * tw_real;
        
        window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Store results
    const int output_idx = global_window * 1024 + point_id;
    output[output_idx] = window_data[window_id][point_id];
}

extern "C" void launch_fft1024_simple(const cuComplex* input, cuComplex* output, int num_windows) {
    static bool initialized = false;
    if (!initialized) {
        init_fft1024_twiddles();
        initialized = true;
    }
    
    dim3 blockDim(1, 1024);
    int num_blocks = num_windows; // 1 window per block
    size_t shared_mem_size = 1 * 1024 * sizeof(cuComplex);
    
    fft1024_simple_kernel<<<num_blocks, blockDim, shared_mem_size>>>(
        input, output, num_windows
    );
}


