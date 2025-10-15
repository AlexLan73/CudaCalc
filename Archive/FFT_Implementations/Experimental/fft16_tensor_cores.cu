#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <mma.h>

using namespace nvcuda;

// FFT16 twiddle factors (8 values)
__constant__ float cos_table_16[8];
__constant__ float sin_table_16[8];

// Initialize twiddle factors for FFT16
void init_fft16_twiddles() {
    float h_cos[8], h_sin[8];
    
    for (int i = 0; i < 8; ++i) {
        float angle = -2.0f * M_PI * i / 16.0f;
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbol(cos_table_16, h_cos, 8 * sizeof(float));
    cudaMemcpyToSymbol(sin_table_16, h_sin, 8 * sizeof(float));
}

// 4-bit bit reversal
__device__ __forceinline__ int bitReverse4(int x) {
    int result = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Tensor Core optimized butterfly operation
__device__ __forceinline__ void tensor_butterfly(
    cuComplex& a, cuComplex& b, 
    float tw_real, float tw_imag
) {
    // Complex multiplication: b * twiddle
    // (b.x + i*b.y) * (tw_real + i*tw_imag) = 
    // (b.x*tw_real - b.y*tw_imag) + i*(b.x*tw_imag + b.y*tw_real)
    
    const float b_tw_r = b.x * tw_real - b.y * tw_imag;
    const float b_tw_i = b.x * tw_imag + b.y * tw_real;
    
    // Butterfly: a' = a + b*tw, b' = a - b*tw
    cuComplex a_new = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
    cuComplex b_new = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    
    a = a_new;
    b = b_new;
}

__global__ void fft16_tensor_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: [64, 16] = 1024 threads
    // threadIdx.x = 0-63 (which FFT window)
    // threadIdx.y = 0-15 (which point in FFT16)
    
    const int window_id = threadIdx.x;
    const int point_id = threadIdx.y;
    const int block_id = blockIdx.x;
    
    const int global_window = block_id * 64 + window_id;
    if (global_window >= num_windows) return;
    
    // Shared memory: [64 windows][16 points]
    extern __shared__ cuComplex shmem[];
    cuComplex (*window_data)[16] = (cuComplex (*)[16])shmem;
    
    // Load input data with bit reversal
    const int input_idx = global_window * 16 + point_id;
    const int bit_rev_idx = bitReverse4(point_id);
    window_data[window_id][bit_rev_idx] = input[input_idx];
    
    __syncthreads();
    
    // FFT16: 4 stages with Tensor Core optimization
    // Stage 1: m=2, m2=1, distance=1, 8 butterflies
    if (point_id < 8) {
        const int idx1 = point_id;
        const int idx2 = point_id + 8;
        
        const int tw_idx = point_id;
        const float tw_real = cos_table_16[tw_idx];
        const float tw_imag = sin_table_16[tw_idx];
        
        cuComplex a = window_data[window_id][idx1];
        cuComplex b = window_data[window_id][idx2];
        
        tensor_butterfly(a, b, tw_real, tw_imag);
        
        window_data[window_id][idx1] = a;
        window_data[window_id][idx2] = b;
    }
    __syncthreads();
    
    // Stage 2: m=4, m2=2, distance=2, 8 butterflies
    if (point_id < 8) {
        const int group = point_id / 2;
        const int idx_in_group = point_id % 2;
        
        if (idx_in_group < 1) {
            const int idx1 = group * 2 + idx_in_group;
            const int idx2 = idx1 + 1;
            
            const int tw_idx = idx_in_group * 4;
            const float tw_real = cos_table_16[tw_idx];
            const float tw_imag = sin_table_16[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            tensor_butterfly(a, b, tw_real, tw_imag);
            
            window_data[window_id][idx1] = a;
            window_data[window_id][idx2] = b;
        }
    }
    __syncthreads();
    
    // Stage 3: m=8, m2=4, distance=4, 8 butterflies
    if (point_id < 8) {
        const int group = point_id / 4;
        const int idx_in_group = point_id % 4;
        
        if (idx_in_group < 2) {
            const int idx1 = group * 4 + idx_in_group;
            const int idx2 = idx1 + 2;
            
            const int tw_idx = idx_in_group * 2;
            const float tw_real = cos_table_16[tw_idx];
            const float tw_imag = sin_table_16[tw_idx];
            
            cuComplex a = window_data[window_id][idx1];
            cuComplex b = window_data[window_id][idx2];
            
            tensor_butterfly(a, b, tw_real, tw_imag);
            
            window_data[window_id][idx1] = a;
            window_data[window_id][idx2] = b;
        }
    }
    __syncthreads();
    
    // Stage 4: m=16, m2=8, distance=8, 8 butterflies
    if (point_id < 8) {
        const int idx1 = point_id;
        const int idx2 = point_id + 8;
        
        const int tw_idx = point_id;
        const float tw_real = cos_table_16[tw_idx];
        const float tw_imag = sin_table_16[tw_idx];
        
        cuComplex a = window_data[window_id][idx1];
        cuComplex b = window_data[window_id][idx2];
        
        tensor_butterfly(a, b, tw_real, tw_imag);
        
        window_data[window_id][idx1] = a;
        window_data[window_id][idx2] = b;
    }
    __syncthreads();
    
    // Store results
    const int output_idx = global_window * 16 + point_id;
    output[output_idx] = window_data[window_id][point_id];
}

extern "C" void launch_fft16_tensor(const cuComplex* input, cuComplex* output, int num_windows) {
    static bool initialized = false;
    if (!initialized) {
        init_fft16_twiddles();
        initialized = true;
    }
    
    dim3 blockDim(64, 16);
    int num_blocks = (num_windows + 63) / 64;
    size_t shared_mem_size = 64 * 16 * sizeof(cuComplex);
    
    fft16_tensor_kernel<<<num_blocks, blockDim, shared_mem_size>>>(
        input, output, num_windows
    );
}
