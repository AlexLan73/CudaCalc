#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// FFT512 twiddle factors (256 values)
__constant__ float cos_table_512[256];
__constant__ float sin_table_512[256];

// Initialize twiddle factors for FFT512
void init_fft512_twiddles() {
    float h_cos[256], h_sin[256];
    
    for (int i = 0; i < 256; ++i) {
        float angle = -2.0f * M_PI * i / 512.0f;
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbol(cos_table_512, h_cos, 256 * sizeof(float));
    cudaMemcpyToSymbol(sin_table_512, h_sin, 256 * sizeof(float));
}

// 9-bit bit reversal
__device__ __forceinline__ int bitReverse9(int x) {
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__global__ void fft512_simple_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: [2, 512] = 1024 threads
    // threadIdx.x = 0-1 (which FFT window)
    // threadIdx.y = 0-511 (which point in FFT512)
    
    const int window_id = threadIdx.x;
    const int point_id = threadIdx.y;
    const int block_id = blockIdx.x;
    
    const int global_window = block_id * 2 + window_id;
    if (global_window >= num_windows) return;
    
    // Shared memory: [2 windows][512 points]
    extern __shared__ cuComplex shmem[];
    cuComplex (*window_data)[512] = (cuComplex (*)[512])shmem;
    
    // Load input data with bit reversal
    const int input_idx = global_window * 512 + point_id;
    const int bit_rev_idx = bitReverse9(point_id);
    window_data[window_id][bit_rev_idx] = input[input_idx];
    
    __syncthreads();
    
    // FFT512: 9 stages
    // Stage 1: m=2, m2=1, distance=1, 256 butterflies
    {
        const int m = 2;
        const int m2 = 1;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (512 / m);
            const float tw_real = cos_table_512[tw_idx];
            const float tw_imag = sin_table_512[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 2: m=4, m2=2, distance=2, 256 butterflies
    {
        const int m = 4;
        const int m2 = 2;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (512 / m);
            const float tw_real = cos_table_512[tw_idx];
            const float tw_imag = sin_table_512[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 3: m=8, m2=4, distance=4, 256 butterflies
    {
        const int m = 8;
        const int m2 = 4;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (512 / m);
            const float tw_real = cos_table_512[tw_idx];
            const float tw_imag = sin_table_512[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 4: m=16, m2=8, distance=8, 256 butterflies
    {
        const int m = 16;
        const int m2 = 8;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (512 / m);
            const float tw_real = cos_table_512[tw_idx];
            const float tw_imag = sin_table_512[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 5: m=32, m2=16, distance=16, 256 butterflies
    {
        const int m = 32;
        const int m2 = 16;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (512 / m);
            const float tw_real = cos_table_512[tw_idx];
            const float tw_imag = sin_table_512[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 6: m=64, m2=32, distance=32, 256 butterflies
    {
        const int m = 64;
        const int m2 = 32;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (512 / m);
            const float tw_real = cos_table_512[tw_idx];
            const float tw_imag = sin_table_512[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 7: m=128, m2=64, distance=64, 256 butterflies
    {
        const int m = 128;
        const int m2 = 64;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (512 / m);
            const float tw_real = cos_table_512[tw_idx];
            const float tw_imag = sin_table_512[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 8: m=256, m2=128, distance=128, 256 butterflies
    {
        const int m = 256;
        const int m2 = 128;
        const int group = point_id / m;
        const int idx_in_group = point_id % m;
        
        if (idx_in_group < m2) {
            const int idx1 = group * m + idx_in_group;
            const int idx2 = idx1 + m2;
            
            const int tw_idx = idx_in_group * (512 / m);
            const float tw_real = cos_table_512[tw_idx];
            const float tw_imag = sin_table_512[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            const float b_tw_r = b.x * tw_real - b.y * tw_imag;
            const float b_tw_i = b.x * tw_imag + b.y * tw_real;
            
            window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
            window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
        }
    }
    __syncthreads();
    
    // Stage 9: m=512, m2=256, distance=256, 256 butterflies
    if (point_id < 256) {
        const int idx1 = point_id;
        const int idx2 = point_id + 256;
        
        const int tw_idx = point_id;
        const float tw_real = cos_table_512[tw_idx];
        const float tw_imag = sin_table_512[tw_idx];
        
        cuComplex a = window_data[window_id][idx1];
        cuComplex b = window_data[window_id][idx2];
        
        const float b_tw_r = b.x * tw_real - b.y * tw_imag;
        const float b_tw_i = b.x * tw_imag + b.y * tw_real;
        
        window_data[window_id][idx1] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        window_data[window_id][idx2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Store results
    const int output_idx = global_window * 512 + point_id;
    output[output_idx] = window_data[window_id][point_id];
}

extern "C" void launch_fft512_simple(const cuComplex* input, cuComplex* output, int num_windows) {
    static bool initialized = false;
    if (!initialized) {
        init_fft512_twiddles();
        initialized = true;
    }
    
    dim3 blockDim(2, 512);
    int num_blocks = (num_windows + 1) / 2;
    size_t shared_mem_size = 2 * 512 * sizeof(cuComplex);
    
    fft512_simple_kernel<<<num_blocks, blockDim, shared_mem_size>>>(
        input, output, num_windows
    );
}


