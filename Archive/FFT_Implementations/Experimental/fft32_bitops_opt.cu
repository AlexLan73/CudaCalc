/**
 * @file fft32_bitops_opt.cu
 * @brief FFT32 with BIT OPERATIONS optimization
 * 
 * Replace division/modulo with bit shifts and masks
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

__constant__ float TWIDDLE_32_COS[16] = {
    1.000000f, 0.980785f, 0.923880f, 0.831470f,
    0.707107f, 0.555570f, 0.382683f, 0.195090f,
    0.000000f, -0.195090f, -0.382683f, -0.555570f,
   -0.707107f, -0.831470f, -0.923880f, -0.980785f
};

__constant__ float TWIDDLE_32_SIN[16] = {
    0.000000f, -0.195090f, -0.382683f, -0.555570f,
   -0.707107f, -0.831470f, -0.923880f, -0.980785f,
   -1.000000f, -0.980785f, -0.923880f, -0.831470f,
   -0.707107f, -0.555570f, -0.382683f, -0.195090f
};

// Optimized bit reverse using bit operations
static __device__ __forceinline__ int bitReverse5_fast(int x) {
    return ((x & 1) << 4) | ((x & 2) << 2) | (x & 4) | ((x & 8) >> 2) | ((x & 16) >> 4);
}

// Complex multiply - inlined
static __device__ __forceinline__ float2 complex_mul(float2 a, float tw_cos, float tw_sin) {
    return make_float2(a.x * tw_cos - a.y * tw_sin, a.x * tw_sin + a.y * tw_cos);
}

__global__ void fft32_bitops_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int global_fft_id = blockIdx.x * 32 + x;
    
    if (global_fft_id >= num_windows) return;
    
    __shared__ float2 shmem[32][34];
    
    // Load with __ldg()
    const int input_idx = global_fft_id * 32 + y;
    const cuComplex val = __ldg(&input[input_idx]);
    shmem[x][bitReverse5_fast(y)] = make_float2(val.x, val.y);
    __syncthreads();
    
    // STAGE 0: m=2, m2=1 - using bit operations
    if (y < 16) {
        const int idx1 = (y & ~1) | (y & 1);        // k + j using bits
        const int idx2 = idx1 + 1;
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        float2 t = complex_mul(v, TWIDDLE_32_COS[0], TWIDDLE_32_SIN[0]);  // Always 1+0i
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 1: m=4, m2=2 - bit ops
    if (y < 16) {
        const int idx1 = (y & ~3) | (y & 1);  // More complex but faster
        const int idx2 = idx1 + 2;
        const int j = y & 1;
        const int tw_idx = j << 3;  // j * 8
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        float2 t = complex_mul(v, TWIDDLE_32_COS[tw_idx], TWIDDLE_32_SIN[tw_idx]);
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 2: m=8, m2=4
    if (y < 16) {
        const int idx1 = (y & ~7) | (y & 3);
        const int idx2 = idx1 + 4;
        const int j = y & 3;
        const int tw_idx = j << 2;  // j * 4
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        float2 t = complex_mul(v, TWIDDLE_32_COS[tw_idx], TWIDDLE_32_SIN[tw_idx]);
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 3: m=16, m2=8
    if (y < 16) {
        const int idx1 = (y & ~15) | (y & 7);
        const int idx2 = idx1 + 8;
        const int j = y & 7;
        const int tw_idx = j << 1;  // j * 2
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        float2 t = complex_mul(v, TWIDDLE_32_COS[tw_idx], TWIDDLE_32_SIN[tw_idx]);
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // STAGE 4: m=32, m2=16 (final)
    if (y < 16) {
        const int idx1 = y;
        const int idx2 = y + 16;
        
        float2 u = shmem[x][idx1];
        float2 v = shmem[x][idx2];
        float2 t = complex_mul(v, TWIDDLE_32_COS[y], TWIDDLE_32_SIN[y]);
        
        shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
        shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
    }
    __syncthreads();
    
    // Store
    const int output_idx = global_fft_id * 32 + y;
    output[output_idx] = make_cuComplex(shmem[x][y].x, shmem[x][y].y);
}

extern "C" void launch_fft32_bitops(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(32, 32);
    int num_blocks = (num_windows + 31) / 32;
    dim3 grid(num_blocks);
    
    fft32_bitops_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}



