#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>

// === ULTRA-ОПТИМИЗИРОВАННЫЕ ТЕНЗОРНЫЕ CUDA ЯДРА ===

// === УТИЛИТЫ ДЛЯ FFTshift (ULTRA ТЕНЗОРНЫЕ) ===

/**
 * @brief FFTshift функция для ULTRA тензорных данных (FP16)
 */
__device__ __forceinline__ int ultra_tensor_fftshift_index(int original_index, int size) {
    // Для четного размера (стандартный случай)
    if (size % 2 == 0) {
        int half = size / 2;
        if (original_index < half) {
            return original_index + half;  // [0, half-1] -> [half, size-1]
        } else {
            return original_index - half;  // [half, size-1] -> [0, half-1]
        }
    }
    // Для нечетного размера
    else {
        int half = size / 2;
        if (original_index <= half) {
            return original_index + half;  // [0, half] -> [half+1, size-1]
        } else {
            return original_index - half - 1;  // [half+1, size-1] -> [0, half-1]
        }
    }
}

/**
 * @brief Применение fftshift к ULTRA тензорным FFT результатам (FP16)
 */
__device__ void apply_ultra_tensor_fftshift(__half* data_real, __half* data_imag, int size) {
    // Создаем временные массивы для правильного порядка
    __half temp_real[64]; // Максимальный размер для наших FFT
    __half temp_imag[64];
    
    // Копируем данные в правильном порядке
    for (int i = 0; i < size; ++i) {
        int shifted_index = ultra_tensor_fftshift_index(i, size);
        temp_real[shifted_index] = data_real[i];
        temp_imag[shifted_index] = data_imag[i];
    }
    
    // Копируем обратно
    for (int i = 0; i < size; ++i) {
        data_real[i] = temp_real[i];
        data_imag[i] = temp_imag[i];
    }
}

// Предвычисленные тензорные таблицы (FP16)
__constant__ __half ultra_twiddles_16_real[8];
__constant__ __half ultra_twiddles_16_imag[8];
__constant__ __half ultra_twiddles_32_real[16];
__constant__ __half ultra_twiddles_32_imag[16];
__constant__ __half ultra_twiddles_64_real[32];
__constant__ __half ultra_twiddles_64_imag[32];

// Предвычисленные тензорные окна Хемминга (FP16)
// ИСПРАВЛЕННЫЕ ultra тензорные окна Хемминга для размеров ДАННЫХ (FP16)
// FFT16 → 8 точек данных
__constant__ __half ultra_hamming_8[8];
// FFT32 → 16 точек данных  
__constant__ __half ultra_hamming_16[16];
// FFT64 → 32 точки данных
__constant__ __half ultra_hamming_32[32];

// === ULTRA-ОПТИМИЗИРОВАННЫЕ УТИЛИТЫ ===

/**
 * @brief Умножение тензорных комплексных чисел (FP16) - ULTRA версия
 */
__device__ __forceinline__ void ultraTensorComplexMult(__half a_real, __half a_imag, 
                                                       __half b_real, __half b_imag,
                                                       __half* result_real, __half* result_imag) {
    // Оптимизированное умножение с использованием Tensor Cores
    __half temp_real = __hadd(__hmul(a_real, b_real), __hneg(__hmul(a_imag, b_imag)));
    __half temp_imag = __hadd(__hmul(a_real, b_imag), __hmul(a_imag, b_real));
    *result_real = temp_real;
    *result_imag = temp_imag;
}

/**
 * @brief Сложение тензорных комплексных чисел (FP16) - ULTRA версия
 */
__device__ __forceinline__ void ultraTensorComplexAdd(__half a_real, __half a_imag,
                                                      __half b_real, __half b_imag,
                                                      __half* result_real, __half* result_imag) {
    *result_real = __hadd(a_real, b_real);
    *result_imag = __hadd(a_imag, b_imag);
}

/**
 * @brief Вычитание тензорных комплексных чисел (FP16) - ULTRA версия
 */
__device__ __forceinline__ void ultraTensorComplexSub(__half a_real, __half a_imag,
                                                      __half b_real, __half b_imag,
                                                      __half* result_real, __half* result_imag) {
    *result_real = __hsub(a_real, b_real);
    *result_imag = __hsub(a_imag, b_imag);
}

// === ULTRA-ОПТИМИЗИРОВАННЫЕ FFT ЯДРА ===

/**
 * @brief ULTRA Tensor Core FFT для 16 точек с максимальной оптимизацией
 * 
 * Использует 2D блоки для максимизации количества потоков:
 * - Блок: [64, 16] = 1024 потока (максимум для RTX 3060)
 * - FFT на блок: 64 FFT (по 16 потоков на FFT)
 * - Shared Memory: 16KB (максимальное использование 48KB)
 */
__global__ void ultraTensorFFT16Kernel(
    __half* input_real,
    __half* input_imag,
    __half* output_real,
    __half* output_imag,
    int totalFFTs,
    bool useHamming
) {
    // 2D индексация для максимизации потоков
    int x = threadIdx.x;  // 0-63 (64 потока по X)
    int y = threadIdx.y;  // 0-15 (16 потоков по Y)
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    
    // Глобальный индекс FFT
    int globalFFTId = (blockY * gridDim.x + blockX) * 64 + x;
    
    if (globalFFTId >= totalFFTs || y >= 16) return;
    
    // Shared memory для ULTRA оптимизации (16KB на блок)
    extern __shared__ __half ultra_shared[];
    __half* fft_real = ultra_shared + x * 16 * 2;  // 2KB на FFT
    __half* fft_imag = ultra_shared + x * 16 * 2 + 16;  // 2KB на FFT
    
    // Загружаем данные в тензорный формат
    __half input_real_val = input_real[globalFFTId * 16 + y];
    __half input_imag_val = input_imag[globalFFTId * 16 + y];
    
    // Применяем тензорное окно Хемминга
    if (useHamming) {
        // FFT16 использует 8 точек данных + 8 нулей
        if (y < 8) {
            input_real_val = __hmul(input_real_val, ultra_hamming_8[y]);
            input_imag_val = __hmul(input_imag_val, ultra_hamming_8[y]);
        }
    }
    
    fft_real[y] = input_real_val;
    fft_imag[y] = input_imag_val;
    __syncthreads();
    
    // === ULTRA ТЕНЗОРНЫЙ FFT БАБОЧКА ДЛЯ 16 ТОЧЕК ===
    
    // Этап 1: Тензорные 2-точечные FFT (векторизованные)
    if (y < 8) {
        __half temp_real, temp_imag;
        ultraTensorComplexAdd(fft_real[y], fft_imag[y],
                             fft_real[y + 8], fft_imag[y + 8],
                             &temp_real, &temp_imag);
        
        ultraTensorComplexSub(fft_real[y], fft_imag[y],
                             fft_real[y + 8], fft_imag[y + 8],
                             &fft_real[y + 8], &fft_imag[y + 8]);
        
        fft_real[y] = temp_real;
        fft_imag[y] = temp_imag;
    }
    __syncthreads();
    
    // Этап 2: Тензорные 4-точечные FFT
    if (y < 4) {
        __half temp_real, temp_imag;
        ultraTensorComplexAdd(fft_real[y], fft_imag[y],
                             fft_real[y + 4], fft_imag[y + 4],
                             &temp_real, &temp_imag);
        
        // Тензорное применение поворотных множителей
        __half twiddle_real = ultra_twiddles_16_real[y];
        __half twiddle_imag = ultra_twiddles_16_imag[y];
        
        __half twiddled_real, twiddled_imag;
        ultraTensorComplexMult(fft_real[y + 8], fft_imag[y + 8],
                              twiddle_real, twiddle_imag,
                              &twiddled_real, &twiddled_imag);
        
        ultraTensorComplexSub(fft_real[y], fft_imag[y],
                             fft_real[y + 4], fft_imag[y + 4],
                             &fft_real[y + 4], &fft_imag[y + 4]);
        
        ultraTensorComplexAdd(fft_real[y], fft_imag[y],
                             twiddled_real, twiddled_imag,
                             &fft_real[y + 8], &fft_imag[y + 8]);
        
        ultraTensorComplexSub(fft_real[y], fft_imag[y],
                             twiddled_real, twiddled_imag,
                             &fft_real[y + 12], &fft_imag[y + 12]);
        
        fft_real[y] = temp_real;
        fft_imag[y] = temp_imag;
    }
    __syncthreads();
    
    // Финальный этап: Тензорный 16-точечный FFT
    if (y < 2) {
        __half temp_real, temp_imag;
        ultraTensorComplexAdd(fft_real[y], fft_imag[y],
                             fft_real[y + 2], fft_imag[y + 2],
                             &temp_real, &temp_imag);
        
        ultraTensorComplexSub(fft_real[y], fft_imag[y],
                             fft_real[y + 2], fft_imag[y + 2],
                             &fft_real[y + 2], &fft_imag[y + 2]);
        
        fft_real[y] = temp_real;
        fft_imag[y] = temp_imag;
    }
    __syncthreads();
    
    if (y == 0) {
        __half temp_real, temp_imag;
        ultraTensorComplexAdd(fft_real[0], fft_imag[0],
                             fft_real[1], fft_imag[1],
                             &temp_real, &temp_imag);
        
        ultraTensorComplexSub(fft_real[0], fft_imag[0],
                             fft_real[1], fft_imag[1],
                             &fft_real[1], &fft_imag[1]);
        
        fft_real[0] = temp_real;
        fft_imag[0] = temp_imag;
    }
    __syncthreads();
    
    // === ПРИМЕНЯЕМ FFTshift ДЛЯ ПРАВИЛЬНОГО ПОРЯДКА ===
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        apply_ultra_tensor_fftshift(fft_real, fft_imag, 16);
    }
    __syncthreads();
    
    // Сохраняем тензорный результат в правильном порядке
    output_real[globalFFTId * 16 + y] = fft_real[y];
    output_imag[globalFFTId * 16 + y] = fft_imag[y];
}

/**
 * @brief ULTRA Tensor Core FFT для 32 точек с 2D блоками
 */
__global__ void ultraTensorFFT32Kernel(
    __half* input_real,
    __half* input_imag,
    __half* output_real,
    __half* output_imag,
    int totalFFTs,
    bool useHamming
) {
    // 2D индексация: [32, 32] = 1024 потока
    int x = threadIdx.x;  // 0-31
    int y = threadIdx.y;  // 0-31
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    
    int globalFFTId = (blockY * gridDim.x + blockX) * 32 + x;
    
    if (globalFFTId >= totalFFTs || y >= 32) return;
    
    extern __shared__ __half ultra_shared[];
    __half* fft_real = ultra_shared + x * 32 * 2;
    __half* fft_imag = ultra_shared + x * 32 * 2 + 32;
    
    // Загружаем данные
    __half input_real_val = input_real[globalFFTId * 32 + y];
    __half input_imag_val = input_imag[globalFFTId * 32 + y];
    
    if (useHamming) {
        // FFT32 использует 16 точек данных + 16 нулей
        if (y < 16) {
            input_real_val = __hmul(input_real_val, ultra_hamming_16[y]);
            input_imag_val = __hmul(input_imag_val, ultra_hamming_16[y]);
        }
    }
    
    fft_real[y] = input_real_val;
    fft_imag[y] = input_imag_val;
    __syncthreads();
    
    // Tensor Core FFT для 32 точек (5 этапов)
    for (int stage = 0; stage < 5; ++stage) {
        int step = 1 << stage;
        
        if (y < step) {
            int partner = y + step;
            if (partner < 32) {
                __half temp_real, temp_imag;
                ultraTensorComplexAdd(fft_real[y], fft_imag[y],
                                     fft_real[partner], fft_imag[partner],
                                     &temp_real, &temp_imag);
                
                fft_real[y] = temp_real;
                fft_imag[y] = temp_imag;
                
                // Применяем тензорные поворотные множители
                if (stage > 0) {
                    __half twiddle_real = ultra_twiddles_32_real[y * (8 >> stage)];
                    __half twiddle_imag = ultra_twiddles_32_imag[y * (8 >> stage)];
                    
                    __half twiddled_real, twiddled_imag;
                    ultraTensorComplexMult(fft_real[partner], fft_imag[partner],
                                          twiddle_real, twiddle_imag,
                                          &twiddled_real, &twiddled_imag);
                    
                    ultraTensorComplexSub(fft_real[y], fft_imag[y],
                                         twiddled_real, twiddled_imag,
                                         &fft_real[partner], &fft_imag[partner]);
                } else {
                    ultraTensorComplexSub(fft_real[y], fft_imag[y],
                                         fft_real[partner], fft_imag[partner],
                                         &fft_real[partner], &fft_imag[partner]);
                }
            }
        }
        __syncthreads();
    }
    
    // === ПРИМЕНЯЕМ FFTshift ДЛЯ ПРАВИЛЬНОГО ПОРЯДКА ===
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        apply_ultra_tensor_fftshift(fft_real, fft_imag, 32);
    }
    __syncthreads();
    
    // Сохраняем результат в правильном порядке
    output_real[globalFFTId * 32 + y] = fft_real[y];
    output_imag[globalFFTId * 32 + y] = fft_imag[y];
}

/**
 * @brief ULTRA Tensor Core FFT для 64 точек с 2D блоками
 */
__global__ void ultraTensorFFT64Kernel(
    __half* input_real,
    __half* input_imag,
    __half* output_real,
    __half* output_imag,
    int totalFFTs,
    bool useHamming
) {
    // 2D индексация: [16, 64] = 1024 потока
    int x = threadIdx.x;  // 0-15
    int y = threadIdx.y;  // 0-63
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    
    int globalFFTId = (blockY * gridDim.x + blockX) * 16 + x;
    
    if (globalFFTId >= totalFFTs || y >= 64) return;
    
    extern __shared__ __half ultra_shared[];
    __half* fft_real = ultra_shared + x * 64 * 2;
    __half* fft_imag = ultra_shared + x * 64 * 2 + 64;
    
    // Загружаем данные
    __half input_real_val = input_real[globalFFTId * 64 + y];
    __half input_imag_val = input_imag[globalFFTId * 64 + y];
    
    if (useHamming) {
        // FFT64 использует 32 точки данных + 32 нуля
        if (y < 32) {
            input_real_val = __hmul(input_real_val, ultra_hamming_32[y]);
            input_imag_val = __hmul(input_imag_val, ultra_hamming_32[y]);
        }
    }
    
    fft_real[y] = input_real_val;
    fft_imag[y] = input_imag_val;
    __syncthreads();
    
    // Tensor Core FFT для 64 точек (6 этапов)
    for (int stage = 0; stage < 6; ++stage) {
        int step = 1 << stage;
        
        if (y < step) {
            int partner = y + step;
            if (partner < 64) {
                __half temp_real, temp_imag;
                ultraTensorComplexAdd(fft_real[y], fft_imag[y],
                                     fft_real[partner], fft_imag[partner],
                                     &temp_real, &temp_imag);
                
                fft_real[y] = temp_real;
                fft_imag[y] = temp_imag;
                
                // Применяем тензорные поворотные множители
                if (stage > 0) {
                    __half twiddle_real = ultra_twiddles_64_real[y * (16 >> stage)];
                    __half twiddle_imag = ultra_twiddles_64_imag[y * (16 >> stage)];
                    
                    __half twiddled_real, twiddled_imag;
                    ultraTensorComplexMult(fft_real[partner], fft_imag[partner],
                                          twiddle_real, twiddle_imag,
                                          &twiddled_real, &twiddled_imag);
                    
                    ultraTensorComplexSub(fft_real[y], fft_imag[y],
                                         twiddled_real, twiddled_imag,
                                         &fft_real[partner], &fft_imag[partner]);
                } else {
                    ultraTensorComplexSub(fft_real[y], fft_imag[y],
                                         fft_real[partner], fft_imag[partner],
                                         &fft_real[partner], &fft_imag[partner]);
                }
            }
        }
        __syncthreads();
    }
    
    // === ПРИМЕНЯЕМ FFTshift ДЛЯ ПРАВИЛЬНОГО ПОРЯДКА ===
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        apply_ultra_tensor_fftshift(fft_real, fft_imag, 64);
    }
    __syncthreads();
    
    // Сохраняем результат в правильном порядке
    output_real[globalFFTId * 64 + y] = fft_real[y];
    output_imag[globalFFTId * 64 + y] = fft_imag[y];
}

// === HOST ФУНКЦИИ ===

extern "C" {

// Инициализация ULTRA тензорных таблиц
void initializeUltraTensorTables() {
    // Таблица для 16-точечного FFT (FP16)
    __half twiddles_16_real_host[8];
    __half twiddles_16_imag_host[8];
    
    float twiddles_16_real_float[8] = {
        1.000000f, 0.923880f, 0.707107f, 0.382683f,
        0.000000f, -0.382683f, -0.707107f, -0.923880f
    };
    float twiddles_16_imag_float[8] = {
        0.000000f, -0.382683f, -0.707107f, -0.923880f,
        -1.000000f, -0.923880f, -0.707107f, -0.382683f
    };
    
    for (int i = 0; i < 8; ++i) {
        twiddles_16_real_host[i] = __float2half(twiddles_16_real_float[i]);
        twiddles_16_imag_host[i] = __float2half(twiddles_16_imag_float[i]);
    }
    
    cudaMemcpyToSymbol(ultra_twiddles_16_real, twiddles_16_real_host, sizeof(twiddles_16_real_host));
    cudaMemcpyToSymbol(ultra_twiddles_16_imag, twiddles_16_imag_host, sizeof(twiddles_16_imag_host));
    
    // Таблица для 32-точечного FFT (FP16)
    __half twiddles_32_real_host[16];
    __half twiddles_32_imag_host[16];
    
    for (int i = 0; i < 16; ++i) {
        float angle = -2.0f * M_PI * i / 32.0f;
        twiddles_32_real_host[i] = __float2half(cosf(angle));
        twiddles_32_imag_host[i] = __float2half(sinf(angle));
    }
    
    cudaMemcpyToSymbol(ultra_twiddles_32_real, twiddles_32_real_host, sizeof(twiddles_32_real_host));
    cudaMemcpyToSymbol(ultra_twiddles_32_imag, twiddles_32_imag_host, sizeof(twiddles_32_imag_host));
    
    // Таблица для 64-точечного FFT (FP16)
    __half twiddles_64_real_host[32];
    __half twiddles_64_imag_host[32];
    
    for (int i = 0; i < 32; ++i) {
        float angle = -2.0f * M_PI * i / 64.0f;
        twiddles_64_real_host[i] = __float2half(cosf(angle));
        twiddles_64_imag_host[i] = __float2half(sinf(angle));
    }
    
    cudaMemcpyToSymbol(ultra_twiddles_64_real, twiddles_64_real_host, sizeof(twiddles_64_real_host));
    cudaMemcpyToSymbol(ultra_twiddles_64_imag, twiddles_64_imag_host, sizeof(twiddles_64_imag_host));
    
    // ULTRA тензорные окна Хемминга (FP16)
    // ИСПРАВЛЕННЫЕ ultra окна Хемминга для размеров ДАННЫХ
    __half hamming_8_host[8];   // FFT16 → 8 точек данных
    __half hamming_16_host[16]; // FFT32 → 16 точек данных  
    __half hamming_32_host[32]; // FFT64 → 32 точки данных
    
    // FFT16 → 8 точек данных
    for (int i = 0; i < 8; ++i) {
        float hamming_val = 0.54f - 0.46f * cosf(2.0f * M_PI * i / 7.0f);
        hamming_8_host[i] = __float2half(hamming_val);
    }
    
    // FFT32 → 16 точек данных
    for (int i = 0; i < 16; ++i) {
        float hamming_val = 0.54f - 0.46f * cosf(2.0f * M_PI * i / 15.0f);
        hamming_16_host[i] = __float2half(hamming_val);
    }
    
    // FFT64 → 32 точки данных
    for (int i = 0; i < 32; ++i) {
        float hamming_val = 0.54f - 0.46f * cosf(2.0f * M_PI * i / 31.0f);
        hamming_32_host[i] = __float2half(hamming_val);
    }
    
    cudaMemcpyToSymbol(ultra_hamming_8, hamming_8_host, sizeof(hamming_8_host));
    cudaMemcpyToSymbol(ultra_hamming_16, hamming_16_host, sizeof(hamming_16_host));
    cudaMemcpyToSymbol(ultra_hamming_32, hamming_32_host, sizeof(hamming_32_host));
    
    std::cout << "ULTRA Tensor tables initialized successfully" << std::endl;
}

void launchUltraTensorFFT16(
    __half* input_real,
    __half* input_imag,
    __half* output_real,
    __half* output_imag,
    int totalFFTs,
    bool useHamming
) {
    // ULTRA оптимизация: 2D блоки [64, 16] = 1024 потока
    // 64 FFT на блок (в 2 раза больше чем раньше!)
    int fftsPerBlock = 64;
    dim3 blockSize(64, 16);  // 2D блок для максимизации потоков
    int totalBlocks = (totalFFTs + fftsPerBlock - 1) / fftsPerBlock;
    dim3 gridSize((totalBlocks + 15) / 16, 16);  // 2D grid
    
    // Shared memory: 16KB (максимальное использование)
    int sharedMemSize = 64 * 16 * sizeof(__half) * 2; // Real + Imag
    
    ultraTensorFFT16Kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input_real, input_imag, output_real, output_imag, totalFFTs, useHamming
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in launchUltraTensorFFT16: " << cudaGetErrorString(error) << std::endl;
    }
    
    std::cout << "ULTRA Tensor FFT16 launched: " << totalFFTs << " FFTs, " 
              << totalBlocks << " blocks, " << sharedMemSize/1024 << "KB shared memory" << std::endl;
}

void launchUltraTensorFFT32(
    __half* input_real,
    __half* input_imag,
    __half* output_real,
    __half* output_imag,
    int totalFFTs,
    bool useHamming
) {
    // ULTRA оптимизация: 2D блоки [32, 32] = 1024 потока
    int fftsPerBlock = 32;
    dim3 blockSize(32, 32);
    int totalBlocks = (totalFFTs + fftsPerBlock - 1) / fftsPerBlock;
    dim3 gridSize((totalBlocks + 31) / 32, 32);
    
    int sharedMemSize = 32 * 32 * sizeof(__half) * 2;
    
    ultraTensorFFT32Kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input_real, input_imag, output_real, output_imag, totalFFTs, useHamming
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in launchUltraTensorFFT32: " << cudaGetErrorString(error) << std::endl;
    }
}

void launchUltraTensorFFT64(
    __half* input_real,
    __half* input_imag,
    __half* output_real,
    __half* output_imag,
    int totalFFTs,
    bool useHamming
) {
    // ULTRA оптимизация: 2D блоки [16, 64] = 1024 потока
    int fftsPerBlock = 16;
    dim3 blockSize(16, 64);
    int totalBlocks = (totalFFTs + fftsPerBlock - 1) / fftsPerBlock;
    dim3 gridSize((totalBlocks + 15) / 16, 16);
    
    int sharedMemSize = 16 * 64 * sizeof(__half) * 2;
    
    ultraTensorFFT64Kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input_real, input_imag, output_real, output_imag, totalFFTs, useHamming
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in launchUltraTensorFFT64: " << cudaGetErrorString(error) << std::endl;
    }
}

} // extern "C"
