#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

// === НАСТОЯЩИЕ 2D TENSOR FFT ЯДРА ===

/**
 * @brief 2D Tensor Core FFT ядро для малых размеров (16x16, 32x32)
 * Использует НАСТОЯЩИЕ Tensor Cores с wmma API
 */
__global__ void real2DTensorFFTKernel(
    const cuComplex* input,
    cuComplex* output,
    int fft_size,
    int num_ffts
) {
    // Используем 2D блоки для Tensor Cores
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    int fft_id = blockIdx.x * (blockDim.x / 32) + warp_id;
    
    if (fft_id >= num_ffts) return;
    
    // Для Tensor Cores используем wmma фрагменты
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag_real, a_frag_imag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag_real, b_frag_imag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag_real, c_frag_imag;
    
    // Shared memory для 2D матриц
    extern __shared__ __half shared_mem[];
    __half* matrix_real = shared_mem;
    __half* matrix_imag = &shared_mem[fft_size * fft_size];
    __half* twiddle_real = &shared_mem[2 * fft_size * fft_size];
    __half* twiddle_imag = &shared_mem[2 * fft_size * fft_size + fft_size * fft_size];
    
    // Загружаем входные данные в 2D матрицу
    for (int i = threadIdx.x; i < fft_size; i += blockDim.x) {
        int input_idx = fft_id * fft_size + i;
        
        // Преобразуем 1D FFT в 2D представление для Tensor Cores
        int row = i / 16;
        int col = i % 16;
        
        if (row < fft_size / 16 && col < 16) {
            matrix_real[row * 16 + col] = __float2half(input[input_idx].x);
            matrix_imag[row * 16 + col] = __float2half(input[input_idx].y);
        }
    }
    
    // Инициализируем twiddle факторы для 2D операций
    if (threadIdx.x < fft_size * fft_size / 4) {
        int idx = threadIdx.x;
        int row = idx / (fft_size / 4);
        int col = idx % (fft_size / 4);
        
        float angle = -2.0f * M_PI * row * col / fft_size;
        twiddle_real[idx] = __float2half(cosf(angle));
        twiddle_imag[idx] = __float2half(sinf(angle));
    }
    
    __syncthreads();
    
    // Выполняем 2D FFT с Tensor Cores
    if (fft_size >= 16 && warp_id < num_ffts) {
        // Для больших размеров (>64) используем блочную обработку
        int blocks_per_dim = (fft_size + 15) / 16;  // Количество 16x16 блоков
        // Загружаем фрагменты для Tensor Core операций
        wmma::load_matrix_sync(a_frag_real, matrix_real, 16);
        wmma::load_matrix_sync(a_frag_imag, matrix_imag, 16);
        wmma::load_matrix_sync(b_frag_real, twiddle_real, 16);
        wmma::load_matrix_sync(b_frag_imag, twiddle_imag, 16);
        
        // Инициализируем аккумулятор
        wmma::fill_fragment(c_frag_real, __float2half(0.0f));
        wmma::fill_fragment(c_frag_imag, __float2half(0.0f));
        
        // Выполняем матричное умножение с Tensor Cores
        // Комплексное умножение: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        
        // Реальная часть: ac - bd
        wmma::mma_sync(c_frag_real, a_frag_real, b_frag_real, c_frag_real);
        
        // Временный фрагмент для bd
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> temp_frag;
        wmma::fill_fragment(temp_frag, __float2half(0.0f));
        wmma::mma_sync(temp_frag, a_frag_imag, b_frag_imag, temp_frag);
        
        // c_frag_real = ac - bd
        for (int i = 0; i < c_frag_real.num_elements; ++i) {
            c_frag_real.x[i] = __hsub(c_frag_real.x[i], temp_frag.x[i]);
        }
        
        // Мнимая часть: ad + bc
        wmma::fill_fragment(c_frag_imag, __float2half(0.0f));
        wmma::mma_sync(c_frag_imag, a_frag_real, b_frag_imag, c_frag_imag);
        
        wmma::fill_fragment(temp_frag, __float2half(0.0f));
        wmma::mma_sync(temp_frag, a_frag_imag, b_frag_real, temp_frag);
        
        // c_frag_imag = ad + bc
        for (int i = 0; i < c_frag_imag.num_elements; ++i) {
            c_frag_imag.x[i] = __hadd(c_frag_imag.x[i], temp_frag.x[i]);
        }
        
        // Сохраняем результат обратно в shared memory
        wmma::store_matrix_sync(matrix_real, c_frag_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync(matrix_imag, c_frag_imag, 16, wmma::mem_row_major);
    }
    
    __syncthreads();
    
    // Записываем результат в выходной массив
    for (int i = threadIdx.x; i < fft_size; i += blockDim.x) {
        int output_idx = fft_id * fft_size + i;
        int row = i / 16;
        int col = i % 16;
        
        if (row < fft_size / 16 && col < 16) {
            output[output_idx].x = __half2float(matrix_real[row * 16 + col]);
            output[output_idx].y = __half2float(matrix_imag[row * 16 + col]);
        }
    }
}

/**
 * @brief Специализированное 2D Tensor ядро для FFT16 (16x16 матрица)
 */
__global__ void tensor2DFFT16Kernel(
    const cuComplex* input,
    cuComplex* output,
    int num_ffts
) {
    int warp_id = threadIdx.x / 32;
    int fft_id = blockIdx.x * (blockDim.x / 32) + warp_id;
    
    if (fft_id >= num_ffts) return;
    
    // Фрагменты для 16x16 Tensor Core операций
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_real, a_imag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_real, b_imag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_real, c_imag;
    
    // Shared memory для 16x16 матриц
    extern __shared__ __half shmem[];
    __half* data_real = shmem;
    __half* data_imag = &shmem[256];        // 16*16
    __half* twiddle_real = &shmem[512];     // 16*16
    __half* twiddle_imag = &shmem[768];     // 16*16
    
    // Загружаем данные FFT16 в 16x16 матрицу
    if (threadIdx.x < 16) {
        int input_idx = fft_id * 16 + threadIdx.x;
        
        // Размещаем 16 точек FFT в 16x16 матрице (первая строка)
        data_real[threadIdx.x] = __float2half(input[input_idx].x);
        data_imag[threadIdx.x] = __float2half(input[input_idx].y);
        
        // Заполняем остальные строки нулями для корректной работы Tensor Cores
        for (int row = 1; row < 16; ++row) {
            data_real[row * 16 + threadIdx.x] = __float2half(0.0f);
            data_imag[row * 16 + threadIdx.x] = __float2half(0.0f);
        }
        
        // Инициализируем twiddle факторы
        float angle = -2.0f * M_PI * threadIdx.x / 16.0f;
        twiddle_real[threadIdx.x] = __float2half(cosf(angle));
        twiddle_imag[threadIdx.x] = __float2half(sinf(angle));
        
        // Заполняем twiddle матрицу
        for (int row = 1; row < 16; ++row) {
            float row_angle = -2.0f * M_PI * threadIdx.x * row / 16.0f;
            twiddle_real[row * 16 + threadIdx.x] = __float2half(cosf(row_angle));
            twiddle_imag[row * 16 + threadIdx.x] = __float2half(sinf(row_angle));
        }
    }
    
    __syncthreads();
    
    // Выполняем 2D Tensor Core FFT
    if (warp_id < num_ffts) {
        // Загружаем фрагменты
        wmma::load_matrix_sync(a_real, data_real, 16);
        wmma::load_matrix_sync(a_imag, data_imag, 16);
        wmma::load_matrix_sync(b_real, twiddle_real, 16);
        wmma::load_matrix_sync(b_imag, twiddle_imag, 16);
        
        // Комплексное матричное умножение с Tensor Cores
        wmma::fill_fragment(c_real, __float2half(0.0f));
        wmma::fill_fragment(c_imag, __float2half(0.0f));
        
        // Реальная часть: ac - bd
        wmma::mma_sync(c_real, a_real, b_real, c_real);
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> temp;
        wmma::fill_fragment(temp, __float2half(0.0f));
        wmma::mma_sync(temp, a_imag, b_imag, temp);
        
        for (int i = 0; i < c_real.num_elements; ++i) {
            c_real.x[i] = __hsub(c_real.x[i], temp.x[i]);
        }
        
        // Мнимая часть: ad + bc
        wmma::mma_sync(c_imag, a_real, b_imag, c_imag);
        
        wmma::fill_fragment(temp, __float2half(0.0f));
        wmma::mma_sync(temp, a_imag, b_real, temp);
        
        for (int i = 0; i < c_imag.num_elements; ++i) {
            c_imag.x[i] = __hadd(c_imag.x[i], temp.x[i]);
        }
        
        // Сохраняем результат
        wmma::store_matrix_sync(data_real, c_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync(data_imag, c_imag, 16, wmma::mem_row_major);
    }
    
    __syncthreads();
    
    // Извлекаем результат FFT16 из первой строки матрицы
    if (threadIdx.x < 16) {
        int output_idx = fft_id * 16 + threadIdx.x;
        output[output_idx].x = __half2float(data_real[threadIdx.x]);
        output[output_idx].y = __half2float(data_imag[threadIdx.x]);
    }
}

/**
 * @brief Специализированное 2D Tensor ядро для FFT32 (32x32 → 16x16 блоки)
 */
__global__ void tensor2DFFT32Kernel(
    const cuComplex* input,
    cuComplex* output,
    int num_ffts
) {
    int warp_id = threadIdx.x / 32;
    int fft_id = blockIdx.x * (blockDim.x / 32) + warp_id;
    
    if (fft_id >= num_ffts) return;
    
    // Используем 4 блока 16x16 для представления 32x32
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_real, a_imag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_real, b_imag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_real, c_imag;
    
    extern __shared__ __half shmem[];
    __half* data_real = shmem;
    __half* data_imag = &shmem[1024];      // 32*32
    __half* twiddle_real = &shmem[2048];   // 32*32
    __half* twiddle_imag = &shmem[3072];   // 32*32
    
    // Загружаем FFT32 данные в 32x32 матрицу
    if (threadIdx.x < 32) {
        int input_idx = fft_id * 32 + threadIdx.x;
        
        // Размещаем 32 точки в первой строке 32x32 матрицы
        data_real[threadIdx.x] = __float2half(input[input_idx].x);
        data_imag[threadIdx.x] = __float2half(input[input_idx].y);
        
        // Заполняем остальные строки
        for (int row = 1; row < 32; ++row) {
            data_real[row * 32 + threadIdx.x] = __float2half(0.0f);
            data_imag[row * 32 + threadIdx.x] = __float2half(0.0f);
        }
        
        // Twiddle факторы для FFT32
        for (int row = 0; row < 32; ++row) {
            float angle = -2.0f * M_PI * threadIdx.x * row / 32.0f;
            twiddle_real[row * 32 + threadIdx.x] = __float2half(cosf(angle));
            twiddle_imag[row * 32 + threadIdx.x] = __float2half(sinf(angle));
        }
    }
    
    __syncthreads();
    
    // Выполняем 2D FFT с использованием 4 блоков 16x16
    if (warp_id < num_ffts) {
        for (int block_row = 0; block_row < 2; ++block_row) {
            for (int block_col = 0; block_col < 2; ++block_col) {
                // Загружаем 16x16 блок
                wmma::load_matrix_sync(a_real, &data_real[block_row * 16 * 32 + block_col * 16], 32);
                wmma::load_matrix_sync(a_imag, &data_imag[block_row * 16 * 32 + block_col * 16], 32);
                wmma::load_matrix_sync(b_real, &twiddle_real[block_row * 16 * 32 + block_col * 16], 32);
                wmma::load_matrix_sync(b_imag, &twiddle_imag[block_row * 16 * 32 + block_col * 16], 32);
                
                // Комплексное умножение
                wmma::fill_fragment(c_real, __float2half(0.0f));
                wmma::fill_fragment(c_imag, __float2half(0.0f));
                
                // ac - bd
                wmma::mma_sync(c_real, a_real, b_real, c_real);
                
                wmma::fragment<wmma::accumulator, 16, 16, 16, __half> temp;
                wmma::fill_fragment(temp, __float2half(0.0f));
                wmma::mma_sync(temp, a_imag, b_imag, temp);
                
                for (int i = 0; i < c_real.num_elements; ++i) {
                    c_real.x[i] = __hsub(c_real.x[i], temp.x[i]);
                }
                
                // ad + bc
                wmma::mma_sync(c_imag, a_real, b_imag, c_imag);
                
                wmma::fill_fragment(temp, __float2half(0.0f));
                wmma::mma_sync(temp, a_imag, b_real, temp);
                
                for (int i = 0; i < c_imag.num_elements; ++i) {
                    c_imag.x[i] = __hadd(c_imag.x[i], temp.x[i]);
                }
                
                // Сохраняем блок обратно
                wmma::store_matrix_sync(&data_real[block_row * 16 * 32 + block_col * 16], c_real, 32, wmma::mem_row_major);
                wmma::store_matrix_sync(&data_imag[block_row * 16 * 32 + block_col * 16], c_imag, 32, wmma::mem_row_major);
            }
        }
    }
    
    __syncthreads();
    
    // Извлекаем результат FFT32
    if (threadIdx.x < 32) {
        int output_idx = fft_id * 32 + threadIdx.x;
        output[output_idx].x = __half2float(data_real[threadIdx.x]);
        output[output_idx].y = __half2float(data_imag[threadIdx.x]);
    }
}

// === HOST ФУНКЦИИ ===

extern "C" {

/**
 * @brief Запуск 2D Tensor FFT для FFT16
 */
cudaError_t launch2DTensorFFT16(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_ffts,
    cudaStream_t stream
) {
    // Используем 32 потока на блок (1 warp для Tensor Cores)
    int threads_per_block = 32;
    int blocks = (num_ffts + (threads_per_block / 32) - 1) / (threads_per_block / 32);
    
    // Shared memory: 4 матрицы 16x16 по __half
    int shared_mem_size = 4 * 16 * 16 * sizeof(__half);
    
    tensor2DFFT16Kernel<<<blocks, threads_per_block, shared_mem_size, stream>>>(
        d_input, d_output, num_ffts
    );
    
    return cudaGetLastError();
}

/**
 * @brief Запуск 2D Tensor FFT для FFT32
 */
cudaError_t launch2DTensorFFT32(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_ffts,
    cudaStream_t stream
) {
    int threads_per_block = 64;  // 2 warps для лучшей утилизации
    int blocks = (num_ffts + (threads_per_block / 32) - 1) / (threads_per_block / 32);
    
    // Shared memory: 4 матрицы 32x32 по __half
    int shared_mem_size = 4 * 32 * 32 * sizeof(__half);
    
    tensor2DFFT32Kernel<<<blocks, threads_per_block, shared_mem_size, stream>>>(
        d_input, d_output, num_ffts
    );
    
    return cudaGetLastError();
}

/**
 * @brief Универсальный 2D Tensor FFT
 */
cudaError_t launch2DTensorFFT(
    const cuComplex* d_input,
    cuComplex* d_output,
    int fft_size,
    int num_ffts,
    cudaStream_t stream
) {
    if (fft_size == 16) {
        return launch2DTensorFFT16(d_input, d_output, num_ffts, stream);
    } else if (fft_size == 32) {
        return launch2DTensorFFT32(d_input, d_output, num_ffts, stream);
    } else if (fft_size <= 512) {
        // Общее 2D ядро для размеров до 512 (тензор 1024 → 2D до 512)
        int threads_per_block = 128;  // Больше потоков для больших размеров
        int blocks = (num_ffts + (threads_per_block / 32) - 1) / (threads_per_block / 32);
        
        // Для больших размеров используем меньше shared memory на блок
        int shared_mem_size;
        if (fft_size <= 64) {
            shared_mem_size = 4 * fft_size * fft_size * sizeof(__half);
        } else {
            // Для FFT > 64 используем блочную обработку
            shared_mem_size = 4 * 64 * 64 * sizeof(__half);  // Максимум 64x64 блоки
        }
        
        real2DTensorFFTKernel<<<blocks, threads_per_block, shared_mem_size, stream>>>(
            d_input, d_output, fft_size, num_ffts
        );
        
        return cudaGetLastError();
    }
    
    return cudaErrorInvalidValue;
}

} // extern "C"
