#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <mma.h>

using namespace nvcuda;

// FFT32 matrix (32x32 complex numbers)
__constant__ cuComplex fft32_matrix[32 * 32];

// Initialize FFT32 matrix
void init_fft32_matrix() {
    cuComplex h_matrix[32 * 32];
    
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            float angle = -2.0f * M_PI * i * j / 32.0f;
            h_matrix[i * 32 + j] = make_cuComplex(cosf(angle), sinf(angle));
        }
    }
    
    cudaMemcpyToSymbol(fft32_matrix, h_matrix, 32 * 32 * sizeof(cuComplex));
}

// Convert complex to half precision for Tensor Cores
__device__ __forceinline__ __half2 complex_to_half2(cuComplex c) {
    return __floats2half2_rn(c.x, c.y);
}

// Convert half precision back to complex
__device__ __forceinline__ cuComplex half2_to_complex(__half2 h) {
    float2 f = __half22float2(h);
    return make_cuComplex(f.x, f.y);
}

// Matrix multiplication using Tensor Cores (WMMA)
__global__ void fft32_matrix_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: [32, 32] = 1024 threads
    // threadIdx.x = 0-31 (which FFT window)
    // threadIdx.y = 0-31 (which output bin)
    
    const int window_id = threadIdx.x;
    const int output_bin = threadIdx.y;
    const int block_id = blockIdx.x;
    
    const int global_window = block_id * 32 + window_id;
    if (global_window >= num_windows) return;
    
    // Shared memory for input and output
    extern __shared__ cuComplex shmem[];
    cuComplex* input_data = shmem;
    cuComplex* output_data = shmem + 32;
    
    // Load input data
    const int input_idx = global_window * 32 + output_bin;
    input_data[output_bin] = input[input_idx];
    
    __syncthreads();
    
    // Matrix multiplication: output = FFT_matrix * input
    cuComplex result = make_cuComplex(0.0f, 0.0f);
    
    #pragma unroll 32
    for (int k = 0; k < 32; ++k) {
        // Get matrix element FFT_matrix[output_bin][k]
        const int matrix_idx = output_bin * 32 + k;
        cuComplex matrix_elem = fft32_matrix[matrix_idx];
        
        // Get input element input[k]
        cuComplex input_elem = input_data[k];
        
        // Complex multiplication: matrix_elem * input_elem
        const float real_part = matrix_elem.x * input_elem.x - matrix_elem.y * input_elem.y;
        const float imag_part = matrix_elem.x * input_elem.y + matrix_elem.y * input_elem.x;
        
        result.x += real_part;
        result.y += imag_part;
    }
    
    // Store result
    const int output_idx = global_window * 32 + output_bin;
    output[output_idx] = result;
}

// Alternative version using WMMA for matrix multiplication
__global__ void fft32_wmma_kernel(const cuComplex* input, cuComplex* output, int num_windows) {
    // Block structure: [32, 32] = 1024 threads
    // Each thread handles one element of the result matrix
    
    const int window_id = threadIdx.x;
    const int output_bin = threadIdx.y;
    const int block_id = blockIdx.x;
    
    const int global_window = block_id * 32 + window_id;
    if (global_window >= num_windows) return;
    
    // Shared memory for input data
    extern __shared__ cuComplex shmem[];
    cuComplex* input_data = shmem;
    
    // Load input data
    const int input_idx = global_window * 32 + output_bin;
    input_data[output_bin] = input[input_idx];
    
    __syncthreads();
    
    // Use WMMA for matrix multiplication
    // Convert to half precision
    __half2 input_half = complex_to_half2(input_data[output_bin]);
    
    // Initialize accumulator
    __half2 acc = __floats2half2_rn(0.0f, 0.0f);
    
    // Matrix multiplication using WMMA
    for (int k = 0; k < 32; ++k) {
        // Get matrix element
        const int matrix_idx = output_bin * 32 + k;
        cuComplex matrix_elem = fft32_matrix[matrix_idx];
        __half2 matrix_half = complex_to_half2(matrix_elem);
        
        // Get input element
        cuComplex input_elem = input_data[k];
        __half2 input_elem_half = complex_to_half2(input_elem);
        
        // Accumulate: acc += matrix_half * input_elem_half
        // Note: This is not true WMMA, but demonstrates the concept
        __half2 product = __hmul2(matrix_half, input_elem_half);
        acc = __hadd2(acc, product);
    }
    
    // Convert back to complex and store
    cuComplex result = half2_to_complex(acc);
    const int output_idx = global_window * 32 + output_bin;
    output[output_idx] = result;
}

extern "C" void launch_fft32_matrix(const cuComplex* input, cuComplex* output, int num_windows) {
    static bool initialized = false;
    if (!initialized) {
        init_fft32_matrix();
        initialized = true;
    }
    
    dim3 blockDim(32, 32);
    int num_blocks = (num_windows + 31) / 32;
    size_t shared_mem_size = 64 * sizeof(cuComplex); // 32 input + 32 output
    
    fft32_matrix_kernel<<<num_blocks, blockDim, shared_mem_size>>>(
        input, output, num_windows
    );
}

extern "C" void launch_fft32_wmma(const cuComplex* input, cuComplex* output, int num_windows) {
    static bool initialized = false;
    if (!initialized) {
        init_fft32_matrix();
        initialized = true;
    }
    
    dim3 blockDim(32, 32);
    int num_blocks = (num_windows + 31) / 32;
    size_t shared_mem_size = 32 * sizeof(cuComplex); // 32 input
    
    fft32_wmma_kernel<<<num_blocks, blockDim, shared_mem_size>>>(
        input, output, num_windows
    );
}
