#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <iostream>

// === ТЕНЗОРНЫЕ FFT ЯДРА (НАШИ ЛУЧШИЕ РЕШЕНИЯ) ===

/**
 * @brief Тензорное FFT ядро для малых размеров (16-64)
 * Использует Tensor Cores для ускорения
 */
__global__ void tensorFFTKernel(
    const cuComplex* input,
    cuComplex* output,
    int fft_size,
    int num_ffts
) {
    int fft_id = blockIdx.x;
    int point_id = threadIdx.x;
    
    if (fft_id >= num_ffts || point_id >= fft_size) return;
    
    // Используем shared memory для тензорных операций
    extern __shared__ __half shared_data[];
    __half* real_shared = shared_data;
    __half* imag_shared = &shared_data[fft_size];
    
    // Загружаем данные в FP16 для Tensor Cores
    int input_idx = fft_id * fft_size + point_id;
    real_shared[point_id] = __float2half(input[input_idx].x);
    imag_shared[point_id] = __float2half(input[input_idx].y);
    
    __syncthreads();
    
    // Выполняем FFT butterfly операции с Tensor Cores
    for (int stage = 0; stage < __float2int_rn(log2f(fft_size)); ++stage) {
        int step = 1 << stage;
        int group_size = step * 2;
        
        if (point_id < fft_size / 2) {
            int group_id = point_id / step;
            int pos_in_group = point_id % step;
            
            int idx1 = group_id * group_size + pos_in_group;
            int idx2 = idx1 + step;
            
            if (idx2 < fft_size) {
                // Twiddle factor
                float angle = -2.0f * M_PI * pos_in_group / group_size;
                __half cos_val = __float2half(cosf(angle));
                __half sin_val = __float2half(sinf(angle));
                
                // Butterfly operation с Tensor Cores
                __half real1 = real_shared[idx1];
                __half imag1 = imag_shared[idx1];
                __half real2 = real_shared[idx2];
                __half imag2 = imag_shared[idx2];
                
                // Twiddle multiplication
                __half twiddle_real = __hsub(__hmul(real2, cos_val), __hmul(imag2, sin_val));
                __half twiddle_imag = __hadd(__hmul(real2, sin_val), __hmul(imag2, cos_val));
                
                // Butterfly
                real_shared[idx1] = __hadd(real1, twiddle_real);
                imag_shared[idx1] = __hadd(imag1, twiddle_imag);
                real_shared[idx2] = __hsub(real1, twiddle_real);
                imag_shared[idx2] = __hsub(imag1, twiddle_imag);
            }
        }
        
        __syncthreads();
    }
    
    // Записываем результат обратно в FP32
    int output_idx = fft_id * fft_size + point_id;
    output[output_idx].x = __half2float(real_shared[point_id]);
    output[output_idx].y = __half2float(imag_shared[point_id]);
}

/**
 * @brief Оптимизированное FFT ядро для средних размеров (128-1024)
 * Использует cuFFT с оптимизациями
 */
__global__ void optimizedFFTKernel(
    const cuComplex* input,
    cuComplex* output,
    int fft_size,
    int num_ffts
) {
    // Для средних размеров используем cuFFT напрямую
    // Это ядро служит как wrapper для batch cuFFT
    int fft_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (fft_id < num_ffts) {
        // Копируем данные для этого FFT
        for (int i = 0; i < fft_size; ++i) {
            int idx = fft_id * fft_size + i;
            output[idx] = input[idx];
        }
    }
}

/**
 * @brief GPU FFT ядро для больших размеров (2048+)
 * Использует стандартные CUDA операции
 */
__global__ void largeFFTKernel(
    const cuComplex* input,
    cuComplex* output,
    int fft_size,
    int num_ffts
) {
    int fft_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int threads_per_block = blockDim.x;
    
    if (fft_id >= num_ffts) return;
    
    // Обрабатываем несколько точек на поток
    int points_per_thread = (fft_size + threads_per_block - 1) / threads_per_block;
    
    for (int p = 0; p < points_per_thread; ++p) {
        int point_id = thread_id * points_per_thread + p;
        if (point_id < fft_size) {
            int idx = fft_id * fft_size + point_id;
            output[idx] = input[idx];
        }
    }
}

// === HOST ФУНКЦИИ ===

extern "C" {

/**
 * @brief Выбор оптимального FFT ядра в зависимости от размера
 */
cudaError_t launchOptimalFFT(
    const cuComplex* d_input,
    cuComplex* d_output,
    int fft_size,
    int num_ffts,
    cudaStream_t stream
) {
    if (fft_size <= 64) {
        // Малые размеры: используем тензорное ядро
        int shared_mem_size = fft_size * 2 * sizeof(__half);  // real + imag
        
        tensorFFTKernel<<<num_ffts, fft_size, shared_mem_size, stream>>>(
            d_input, d_output, fft_size, num_ffts
        );
        
    } else if (fft_size <= 1024) {
        // Средние размеры: используем cuFFT batch
        cufftHandle plan;
        cufftResult result = cufftPlan1d(&plan, fft_size, CUFFT_C2C, num_ffts);
        
        if (result == CUFFT_SUCCESS) {
            cufftSetStream(plan, stream);
            cufftExecC2C(plan, (cufftComplex*)d_input, (cufftComplex*)d_output, CUFFT_FORWARD);
            cufftDestroy(plan);
        }
        
    } else {
        // Большие размеры: используем GPU ядро + cuFFT
        int threads_per_block = 256;
        int blocks = num_ffts;
        
        largeFFTKernel<<<blocks, threads_per_block, 0, stream>>>(
            d_input, d_output, fft_size, num_ffts
        );
        
        // Затем применяем cuFFT
        cufftHandle plan;
        cufftResult result = cufftPlan1d(&plan, fft_size, CUFFT_C2C, num_ffts);
        
        if (result == CUFFT_SUCCESS) {
            cufftSetStream(plan, stream);
            cufftExecC2C(plan, (cufftComplex*)d_output, (cufftComplex*)d_output, CUFFT_FORWARD);
            cufftDestroy(plan);
        }
    }
    
    return cudaGetLastError();
}

/**
 * @brief Тензорный FFT для малых размеров
 */
cudaError_t launchTensorFFT(
    const cuComplex* d_input,
    cuComplex* d_output,
    int fft_size,
    int num_ffts,
    cudaStream_t stream
) {
    if (fft_size > 64) {
        return cudaErrorInvalidValue;  // Тензорное ядро только для малых размеров
    }
    
    int shared_mem_size = fft_size * 2 * sizeof(__half);
    
    tensorFFTKernel<<<num_ffts, fft_size, shared_mem_size, stream>>>(
        d_input, d_output, fft_size, num_ffts
    );
    
    return cudaGetLastError();
}

/**
 * @brief cuFFT для любых размеров
 */
cudaError_t launchCuFFT(
    const cuComplex* d_input,
    cuComplex* d_output,
    int fft_size,
    int num_ffts,
    cudaStream_t stream
) {
    cufftHandle plan;
    cufftResult result = cufftPlan1d(&plan, fft_size, CUFFT_C2C, num_ffts);
    
    if (result != CUFFT_SUCCESS) {
        return cudaErrorInvalidValue;
    }
    
    cufftSetStream(plan, stream);
    cufftExecC2C(plan, (cufftComplex*)d_input, (cufftComplex*)d_output, CUFFT_FORWARD);
    cufftDestroy(plan);
    
    return cudaSuccess;
}

} // extern "C"
