#include <cuda_runtime.h>
#include <cuComplex.h>

// Наши существующие twiddle factors для FFT16
__constant__ cuComplex fft16_twiddles[8] = {
    {1.000000f, 0.000000f}, {0.923880f, -0.382683f}, {0.707107f, -0.707107f}, {0.382683f, -0.923880f},
    {0.000000f, -1.000000f}, {-0.382683f, -0.923880f}, {-0.707107f, -0.707107f}, {-0.923880f, -0.382683f}
};

// Bit reversal для 4 бит (FFT16)
__device__ __forceinline__ int bit_reverse_4(int x) {
    x = ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    return x >> 12;  // Keep only 4 bits
}

// Комплексное умножение
__device__ __forceinline__ cuComplex complex_multiply(cuComplex a, cuComplex b) {
    cuComplex result;
    result.x = a.x * b.x - a.y * b.y;
    result.y = a.x * b.y + a.y * b.x;
    return result;
}

// Butterfly операция в shared memory
__device__ __forceinline__ void butterfly_shared(
    cuComplex* shared_data,
    int idx1,
    int idx2,
    cuComplex twiddle
) {
    cuComplex a = shared_data[idx1];
    cuComplex b = shared_data[idx2];
    
    // Комплексное умножение: b * twiddle
    cuComplex b_tw = complex_multiply(b, twiddle);
    
    // Butterfly: a' = a + b*tw, b' = a - b*tw
    shared_data[idx1].x = a.x + b_tw.x;
    shared_data[idx1].y = a.y + b_tw.y;
    shared_data[idx2].x = a.x - b_tw.x;
    shared_data[idx2].y = a.y - b_tw.y;
}

// FFT16 в shared memory с overlap-and-save
__device__ void perform_fft16_shared_memory(cuComplex* shared_data, cuComplex* shared_twiddles) {
    // Bit reversal уже выполнен при загрузке, начинаем сразу с FFT
    int tid = threadIdx.x;
    
    // Stage 1: distance 8
    if (tid < 8) {
        butterfly_shared(shared_data, tid, tid + 8, shared_twiddles[0]);
    }
    __syncthreads();
    
    // Stage 2: distance 4
    if (tid < 8) {
        int twiddle_idx = (tid % 2) * 4;
        butterfly_shared(shared_data, tid, tid + 4, shared_twiddles[twiddle_idx]);
    }
    __syncthreads();
    
    // Stage 3: distance 2
    if (tid < 8) {
        int twiddle_idx = (tid % 4) * 2;
        butterfly_shared(shared_data, tid, tid + 2, shared_twiddles[twiddle_idx]);
    }
    __syncthreads();
    
    // Stage 4: distance 1
    if (tid < 8) {
        int twiddle_idx = tid % 8;
        butterfly_shared(shared_data, tid, tid + 1, shared_twiddles[twiddle_idx]);
    }
    __syncthreads();
}

// Основной kernel для FFT16 Shared Memory
__global__ void fft16_shared_memory_kernel(
    const cuComplex* input,
    cuComplex* output,
    int num_windows
) {
    // Shared memory для одного FFT16 (16 × 8 bytes = 128 bytes)
    __shared__ cuComplex shared_data[16];
    // Shared memory для twiddle factors (8 × 8 bytes = 64 bytes)
    __shared__ cuComplex shared_twiddles[8];
    
    int global_window = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_window >= num_windows) return;
    
    int tid = threadIdx.x;
    
    // 0. Копирование twiddle factors в shared memory (один раз на блок)
    if (tid < 8) {
        shared_twiddles[tid] = fft16_twiddles[tid];
    }
    __syncthreads();
    
    // 1. Загрузка данных в shared memory с bit reversal (один раз)
    if (tid < 16) {
        int bit_rev_idx = bit_reverse_4(tid);
        shared_data[tid] = input[global_window * 16 + bit_rev_idx];
    }
    __syncthreads();
    
    // 2. FFT в shared memory (overlap-and-save)
    perform_fft16_shared_memory(shared_data, shared_twiddles);
    
    // 3. Сохранение результатов (один раз)
    if (tid < 16) {
        output[global_window * 16 + tid] = shared_data[tid];
    }
}

// Launch функция
extern "C" void launch_fft16_shared_memory(const cuComplex* input, cuComplex* output, int num_windows) {
    // Оптимальная конфигурация: 8 threads per block для FFT16
    dim3 blockDim(16);  // 16 threads для загрузки/сохранения
    dim3 gridDim((num_windows + blockDim.x - 1) / blockDim.x);
    
    fft16_shared_memory_kernel<<<gridDim, blockDim>>>(input, output, num_windows);
}
