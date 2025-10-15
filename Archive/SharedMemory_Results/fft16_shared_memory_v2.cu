#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>

// Pre-computed twiddle factors for FFT16 (копируем из рабочего алгоритма)
__constant__ float TWIDDLE_16_COS[8] = {
    1.000000f,   // k=0
    0.923880f,   // k=1
    0.707107f,   // k=2
    0.382683f,   // k=3
    0.000000f,   // k=4
   -0.382683f,   // k=5
   -0.707107f,   // k=6
   -0.923880f    // k=7
};

__constant__ float TWIDDLE_16_SIN[8] = {
    0.000000f,   // k=0
   -0.382683f,   // k=1
   -0.707107f,   // k=2
   -0.923880f,   // k=3
   -1.000000f,   // k=4
   -0.923880f,   // k=5
   -0.707107f,   // k=6
   -0.382683f    // k=7
};

// Bit reverse for 4 bits (копируем из рабочего алгоритма)
__device__ int bitReverse4_shared(int x) {
    int result = 0;
    result |= (x & 1) << 3;
    result |= (x & 2) << 1;
    result |= (x & 4) >> 1;
    result |= (x & 8) >> 3;
    return result;
}

// FFT16 в shared memory - копируем рабочий алгоритм
__global__ void fft16_shared_memory_v2_kernel(
    const cuComplex* input,
    cuComplex* output,
    int num_windows
) {
    // Shared memory для одного FFT16
    __shared__ cuComplex shared_data[16];
    
    int global_window = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_window >= num_windows) return;
    
    int tid = threadIdx.x;
    
    // 1. Загрузка данных в shared memory с bit reversal
    if (tid < 16) {
        int bit_rev_idx = bitReverse4_shared(tid);
        shared_data[tid] = input[global_window * 16 + bit_rev_idx];
    }
    __syncthreads();
    
    // 2. FFT в shared memory - копируем рабочий алгоритм
    // Stage 1: distance 8
    if (tid < 8) {
        cuComplex a = shared_data[tid];
        cuComplex b = shared_data[tid + 8];
        
        // Butterfly: a' = a + b, b' = a - b
        shared_data[tid] = make_cuComplex(a.x + b.x, a.y + b.y);
        shared_data[tid + 8] = make_cuComplex(a.x - b.x, a.y - b.y);
    }
    __syncthreads();
    
    // Stage 2: distance 4 (m=4, j=0..1)
    if (tid < 8) {
        cuComplex a = shared_data[tid];
        cuComplex b = shared_data[tid + 4];
        
        // Twiddle factor: W_16^((j * 16) / m) = W_16^((tid/4 * 16) / 4) = W_16^(tid)
        int j = tid / 4;  // j = 0 or 1
        int twiddle_idx = (j * 16) / 4;  // 0 or 4
        float tw_cos = TWIDDLE_16_COS[twiddle_idx % 8];
        float tw_sin = TWIDDLE_16_SIN[twiddle_idx % 8];
        
        // Complex multiplication: b * twiddle
        float b_tw_r = b.x * tw_cos - b.y * tw_sin;
        float b_tw_i = b.x * tw_sin + b.y * tw_cos;
        
        // Butterfly: a' = a + b*tw, b' = a - b*tw
        shared_data[tid] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        shared_data[tid + 4] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Stage 3: distance 2 (m=2, j=0..3)
    if (tid < 8) {
        cuComplex a = shared_data[tid];
        cuComplex b = shared_data[tid + 2];
        
        // Twiddle factor: W_16^((j * 16) / m) = W_16^((tid/2 * 16) / 2) = W_16^(tid*4)
        int j = tid / 2;  // j = 0,1,2,3
        int twiddle_idx = (j * 16) / 2;  // 0,8,16,24 -> 0,0,0,0 (mod 8)
        float tw_cos = TWIDDLE_16_COS[twiddle_idx % 8];
        float tw_sin = TWIDDLE_16_SIN[twiddle_idx % 8];
        
        // Complex multiplication: b * twiddle
        float b_tw_r = b.x * tw_cos - b.y * tw_sin;
        float b_tw_i = b.x * tw_sin + b.y * tw_cos;
        
        // Butterfly: a' = a + b*tw, b' = a - b*tw
        shared_data[tid] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        shared_data[tid + 2] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // Stage 4: distance 1 (m=1, j=0..7)
    if (tid < 8) {
        cuComplex a = shared_data[tid];
        cuComplex b = shared_data[tid + 1];
        
        // Twiddle factor: W_16^((j * 16) / m) = W_16^(tid * 16) = W_16^(tid*16)
        int j = tid;  // j = 0..7
        int twiddle_idx = (j * 16) / 1;  // 0,16,32,48,64,80,96,112 -> 0,0,0,0,0,0,0,0 (mod 8)
        float tw_cos = TWIDDLE_16_COS[twiddle_idx % 8];
        float tw_sin = TWIDDLE_16_SIN[twiddle_idx % 8];
        
        // Complex multiplication: b * twiddle
        float b_tw_r = b.x * tw_cos - b.y * tw_sin;
        float b_tw_i = b.x * tw_sin + b.y * tw_cos;
        
        // Butterfly: a' = a + b*tw, b' = a - b*tw
        shared_data[tid] = make_cuComplex(a.x + b_tw_r, a.y + b_tw_i);
        shared_data[tid + 1] = make_cuComplex(a.x - b_tw_r, a.y - b_tw_i);
    }
    __syncthreads();
    
    // 3. Сохранение результатов
    if (tid < 16) {
        output[global_window * 16 + tid] = shared_data[tid];
    }
}

// Launch функция
extern "C" void launch_fft16_shared_memory_v2(const cuComplex* input, cuComplex* output, int num_windows) {
    dim3 blockDim(16);  // 16 threads per block
    dim3 gridDim((num_windows + blockDim.x - 1) / blockDim.x);
    
    fft16_shared_memory_v2_kernel<<<gridDim, blockDim>>>(input, output, num_windows);
}
