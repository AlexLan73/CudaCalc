/**
 * @file fft32_shared_memory_batch32.cu
 * @brief FFT32 Shared Memory batch - 32 FFT per block with shared memory optimization
 * 
 * Правильная концепция: 32 FFT в блоке [32, 32] с shared memory оптимизацией
 * Сравнивается с fft32_batch32_v2.cu
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

// Pre-computed twiddle factors (same as working version)
__constant__ float TWIDDLE_32_COS_BATCH[16] = {
    1.000000f, 0.980785f, 0.923880f, 0.831470f,
    0.707107f, 0.555570f, 0.382683f, 0.195090f,
    0.000000f, -0.195090f, -0.382683f, -0.555570f,
   -0.707107f, -0.831470f, -0.923880f, -0.980785f
};

__constant__ float TWIDDLE_32_SIN_BATCH[16] = {
    0.000000f, -0.195090f, -0.382683f, -0.555570f,
   -0.707107f, -0.831470f, -0.923880f, -0.980785f,
   -1.000000f, -0.980785f, -0.923880f, -0.831470f,
   -0.707107f, -0.555570f, -0.382683f, -0.195090f
};

__device__ int bitReverse5_batch(int x) {
    int result = 0;
    result |= (x & 1) << 4;
    result |= (x & 2) << 2;
    result |= (x & 4);
    result |= (x & 8) >> 2;
    result |= (x & 16) >> 4;
    return result;
}

__global__ void fft32_shared_memory_batch32_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    // 2D indexing - точно как в batch версии
    const int x = threadIdx.x;  // 0-31: which FFT
    const int tid = threadIdx.y;  // 0-31: which point
    const int fft_id = blockIdx.x * 32 + x;
    
    if (fft_id >= num_windows) return;
    
    // Shared memory: [32 FFT][33 points] + padding для избежания bank conflicts
    __shared__ float2 data[32][33];
    
    // Load with bit-reversal - точно как в batch версии
    const int input_idx = fft_id * 32 + tid;
    const int reversed_tid = bitReverse5_batch(tid);
    data[x][reversed_tid] = make_float2(input[input_idx].x, input[input_idx].y);
    __syncthreads();
    
    // STAGE 0: m=2, m2=1 - точно как в batch версии
    if (tid < 16) {
        const int m2 = 1;
        const int k = (tid / m2) * 2;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int twiddle_idx = j * 16;
        const float tw_cos = __ldg(&TWIDDLE_32_COS_BATCH[twiddle_idx & 15]);
        const float tw_sin = __ldg(&TWIDDLE_32_SIN_BATCH[twiddle_idx & 15]);
        
        float2 u = data[x][idx1];
        float2 v = data[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        data[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        data[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 1: m=4, m2=2 - точно как в batch версии
    if (tid < 16) {
        const int m2 = 2;
        const int k = (tid / m2) * 4;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int twiddle_idx = j * 8;
        const float tw_cos = __ldg(&TWIDDLE_32_COS_BATCH[twiddle_idx]);
        const float tw_sin = __ldg(&TWIDDLE_32_SIN_BATCH[twiddle_idx]);
        
        float2 u = data[x][idx1];
        float2 v = data[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        data[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        data[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 2: m=8, m2=4 - точно как в batch версии
    if (tid < 16) {
        const int m2 = 4;
        const int k = (tid / m2) * 8;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int twiddle_idx = j * 4;
        const float tw_cos = __ldg(&TWIDDLE_32_COS_BATCH[twiddle_idx]);
        const float tw_sin = __ldg(&TWIDDLE_32_SIN_BATCH[twiddle_idx]);
        
        float2 u = data[x][idx1];
        float2 v = data[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        data[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        data[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 3: m=16, m2=8 - точно как в batch версии
    if (tid < 16) {
        const int m2 = 8;
        const int k = (tid / m2) * 16;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int twiddle_idx = j * 2;
        const float tw_cos = __ldg(&TWIDDLE_32_COS_BATCH[twiddle_idx]);
        const float tw_sin = __ldg(&TWIDDLE_32_SIN_BATCH[twiddle_idx]);
        
        float2 u = data[x][idx1];
        float2 v = data[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        data[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        data[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 4: m=32, m2=16 - точно как в batch версии
    if (tid < 16) {
        const int m2 = 16;
        const int k = (tid / m2) * 32;
        const int j = tid % m2;
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        const int twiddle_idx = j * 1;
        const float tw_cos = __ldg(&TWIDDLE_32_COS_BATCH[twiddle_idx]);
        const float tw_sin = __ldg(&TWIDDLE_32_SIN_BATCH[twiddle_idx]);
        
        float2 u = data[x][idx1];
        float2 v = data[x][idx2];
        
        float2 t;
        t.x = v.x * tw_cos - v.y * tw_sin;
        t.y = v.x * tw_sin + v.y * tw_cos;
        
        data[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        data[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // Store result - точно как в batch версии
    const int output_idx = fft_id * 32 + tid;
    output[output_idx] = make_cuComplex(data[x][tid].x, data[x][tid].y);
}

// Host launcher
extern "C" void launch_fft32_shared_memory_batch32(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(32, 32);  // 32 FFT, 32 points per FFT
    dim3 grid((num_windows + 31) / 32);  // Number of blocks needed
    
    fft32_shared_memory_batch32_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}
