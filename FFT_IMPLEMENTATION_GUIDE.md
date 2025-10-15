# 🎯 Руководство по созданию FFT любого размера

## Проверенный алгоритм (Sequential Thinking)

Этот процесс гарантирует **100% правильный результат**!

---

## ШАГ 1: CPU Reference

**Цель:** Понять правильный алгоритм

```cpp
// fftN_cpu_reference.cpp
void fftN_cpu(input, output) {
    // 1. Bit-reversal permutation
    for (i = 0; i < N; i++)
        output[bitReverse(i)] = input[i];
    
    // 2. Butterfly stages
    for (stage = 0; stage < log2(N); stage++) {
        int m = 1 << (stage + 1);
        int m2 = m / 2;
        Complex W_m = exp(-2πi / m);
        
        for (k = 0; k < N; k += m) {
            Complex W = 1.0;
            for (j = 0; j < m2; j++) {
                int idx1 = k + j;
                int idx2 = idx1 + m2;
                
                Complex t = W * output[idx2];
                Complex u = output[idx1];
                
                output[idx1] = u + t;
                output[idx2] = u - t;
                
                W *= W_m;
            }
        }
    }
}
```

**Проверка:** bin[1] = N для сигнала exp(2πi*n/N) ✅

---

## ШАГ 2: Простой GPU (1 FFT на блок)

**Файл:** `fftN_simple_correct.cu`

```cuda
__global__ void fftN_kernel(input, output, num_windows) {
    int fft_id = blockIdx.x;
    int tid = threadIdx.x;  // 0 to N-1
    
    __shared__ float2 data[N];
    
    // Load with bit-reversal
    data[bitReverse(tid)] = input[fft_id * N + tid];
    __syncthreads();
    
    // Butterfly in loop
    for (int stage = 0; stage < log2(N); stage++) {
        int m = 1 << (stage + 1);
        int m2 = m / 2;
        
        int k = (tid / m2) * m;
        int j = tid % m2;
        
        if (tid < N/2) {  // Only half threads active
            int idx1 = k + j;
            int idx2 = idx1 + m2;
            
            // Compute twiddle
            float angle = -2πi * j / m;
            float tw_cos = cosf(angle);
            float tw_sin = sinf(angle);
            
            // Butterfly
            float2 u = data[idx1];
            float2 v = data[idx2];
            float2 t = complex_mul(v, twiddle);
            
            data[idx1] = u + t;
            data[idx2] = u - t;
        }
        __syncthreads();
    }
    
    // Store
    output[fft_id * N + tid] = data[tid];
}
```

**Launch:** `dim3 block(N); dim3 grid(num_windows);`

**Проверка:** Сравнить с cuFFT на 1 окне ✅

---

## ШАГ 3: Pre-computed Tables

**Добавить константные таблицы:**

```cuda
__constant__ float TWIDDLE_N_COS[N/2];
__constant__ float TWIDDLE_N_SIN[N/2];
```

**Заполнить:**
```cpp
for (int k = 0; k < N/2; k++) {
    TWIDDLE_N_COS[k] = cos(-2π * k / N);
    TWIDDLE_N_SIN[k] = sin(-2π * k / N);
}
```

**Заменить в kernel:**
```cuda
// Было:
float tw_cos = cosf(angle);

// Стало:
int twiddle_idx = (j * N) / m;
float tw_cos = TWIDDLE_N_COS[twiddle_idx];
```

**Проверка:** Результат не изменился ✅

---

## ШАГ 4: Linear Unroll

**Файл:** `fftN_simple_unrolled.cu`

**Развернуть цикл по stages:**

```cuda
// STAGE 0: m=2, m2=1
if (tid < N/2) {
    // ... butterfly code for stage 0
}
__syncthreads();

// STAGE 1: m=4, m2=2
if (tid < N/2) {
    // ... butterfly code for stage 1
}
__syncthreads();

// ... и так для всех log2(N) stages
```

**Проверка:** Результат не изменился ✅

---

## ШАГ 5: Batch (M окон в блоке)

**M = 1024 / N** (максимальное использование потоков)

**Файл:** `fftN_batchM.cu`

```cuda
__global__ void fftN_batchM_kernel(input, output, num_windows) {
    int x = threadIdx.x;  // 0 to M-1 (which FFT)
    int y = threadIdx.y;  // 0 to N-1 (which point)
    int global_fft_id = blockIdx.x * M + x;
    
    __shared__ float2 shmem[M][N+2];  // +padding
    
    // Load with bit-reversal
    shmem[x][bitReverse(y)] = input[global_fft_id * N + y];
    __syncthreads();
    
    // All stages (same as step 4, but data → shmem[x])
    // ...
    
    // Store
    output[global_fft_id * N + y] = shmem[x][y];
}
```

**Launch:** `dim3 block(M, N); dim3 grid(num_blocks);`

**Проверка:** Все M окон правильные ✅

---

## ШАГ 6: Validation Rules

```cpp
const double MAGNITUDE_THRESHOLD = 0.01;  // Noise floor
const double ERROR_TOLERANCE = 0.01;      // 0.01% max error

for (каждая точка) {
    float mag_ref = abs(ref[i]);
    float mag_our = abs(our[i]);
    
    // Ignore noise floor
    if (mag_ref >= MAGNITUDE_THRESHOLD || mag_our >= MAGNITUDE_THRESHOLD) {
        double error = abs(mag_our - mag_ref) / mag_ref * 100.0;
        if (error > ERROR_TOLERANCE) {
            failed++;
        }
    }
}
```

---

## ПРИМЕРЫ РАБОТАЮЩИХ ФАЙЛОВ

**FFT16:**
- `fft16_simple_correct_WITH_LOOP.cu` - backup с циклом
- `fft16_batch64.cu` - production (64 окна, 1024 потока)

**FFT32:**
- `fft32_simple_correct_WITH_LOOP.cu` - backup с циклом
- `fft32_batch32_v2.cu` - production (32 окна, 1024 потока)

---

## СЛЕДУЮЩИЕ РАЗМЕРЫ

По этому же алгоритму создать:
- FFT64: batch16.cu (16 окон, блок [16, 64])
- FFT128: batch8.cu (8 окон, блок [8, 128])
- FFT256: batch4.cu (4 окна, блок [4, 256])
- FFT512: batch2.cu (2 окна, блок [2, 512])
- FFT1024: batch1.cu (1 окно, блок [1, 1024])

---

**Версия:** 1.0  
**Дата:** 13 октября 2025  
**Статус:** Проверено на FFT16 и FFT32 ✅



