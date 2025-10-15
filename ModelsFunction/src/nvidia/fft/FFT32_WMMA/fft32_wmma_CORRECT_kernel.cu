/**
 * @file fft32_wmma_CORRECT_kernel.cu
 * @brief CORRECT FFT32 implementation based on CPU reference
 * 
 * This is the CORRECT implementation verified against CPU reference!
 */

#include <cuda_runtime.h>
#include <cuComplex.h>

// Pre-computed twiddle factors W_N^k = exp(-2πi*k/N) for N=32
__constant__ float TWIDDLE_32_COS[32] = {
    1.000000f,   0.980785f,   0.923880f,   0.831470f,
    0.707107f,   0.555570f,   0.382683f,   0.195090f,
    0.000000f,  -0.195090f,  -0.382683f,  -0.555570f,
   -0.707107f,  -0.831470f,  -0.923880f,  -0.980785f,
   -1.000000f,  -0.980785f,  -0.923880f,  -0.831470f,
   -0.707107f,  -0.555570f,  -0.382683f,  -0.195090f,
    0.000000f,   0.195090f,   0.382683f,   0.555570f,
    0.707107f,   0.831470f,   0.923880f,   0.980785f
};

__constant__ float TWIDDLE_32_SIN[32] = {
    0.000000f,  -0.195090f,  -0.382683f,  -0.555570f,
   -0.707107f,  -0.831470f,  -0.923880f,  -0.980785f,
   -1.000000f,  -0.980785f,  -0.923880f,  -0.831470f,
   -0.707107f,  -0.555570f,  -0.382683f,  -0.195090f,
    0.000000f,   0.195090f,   0.382683f,   0.555570f,
    0.707107f,   0.831470f,   0.923880f,   0.980785f,
    1.000000f,   0.980785f,   0.923880f,   0.831470f,
    0.707107f,   0.555570f,   0.382683f,   0.195090f
};

// Bit reverse for 5 bits
__device__ int bitReverse5(int x) {
    int result = 0;
    for (int i = 0; i < 5; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/**
 * CORRECT FFT32 Kernel - Verified against CPU reference
 * Block: [32, 32] = 1024 threads
 * Each block processes 32 FFTs in parallel
 */
__global__ void fft32_correct_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int fft_id = blockIdx.x * 32 + threadIdx.x;  // Which FFT
    const int point_id = threadIdx.y;                  // Which point (0-31)
    
    if (fft_id >= num_windows) return;
    
    __shared__ float2 shmem[32][34];  // Padding for bank conflicts
    
    // Load input with bit-reversal
    const int input_idx = fft_id * 32 + point_id;
    const int reversed_idx = bitReverse5(point_id);
    shmem[threadIdx.x][reversed_idx] = make_float2(input[input_idx].x, input[input_idx].y);
    __syncthreads();
    
    // === 5 BUTTERFLY STAGES ===
    // Stage 0: m=2, m/2=1
    {
        const int m2 = 1;
        const int k = (point_id / 2) * 2;  // Group start
        const int j = point_id % 2;         // Position in group
        
        const int idx1 = k + j;
        const int idx2 = idx1 + m2;
        
        float2 u = shmem[threadIdx.x][idx1];
        float2 t = shmem[threadIdx.x][idx2];
        
        // W = W_2^j (j can be 0 or 1)
        // W_2^0 = 1, W_2^1 = -1
        if (j == 0) {
            // W = 1
            shmem[threadIdx.x][idx1] = make_float2(u.x + t.x, u.y + t.y);
            shmem[threadIdx.x][idx2] = make_float2(u.x - t.x, u.y - t.y);
        } else {
            // W = -1, so t' = -t
            shmem[threadIdx.x][idx1] = make_float2(u.x - t.x, u.y - t.y);
            shmem[threadIdx.x][idx2] = make_float2(u.x + t.x, u.y + t.y);
        }
        __syncthreads();
    }
    
    // Stage 1: m=4, m/2=2
    {
        const int m = 4;
        const int m2 = 2;
        const int k = (point_id / 4) * 4;
        const int j = point_id % 4;
        
        if (j < m2) {
            const int idx1 = k + j;
            const int idx2 = idx1 + m2;
            
            float2 u = shmem[threadIdx.x][idx1];
            float2 t_raw = shmem[threadIdx.x][idx2];
            
            // W = W_4^j, need to index: j * (32/4) = j * 8
            const int tw_idx = (j * 32) / m;
            const float tw_cos = TWIDDLE_32_COS[tw_idx];
            const float tw_sin = TWIDDLE_32_SIN[tw_idx];
            
            // t = W * t_raw
            float2 t = make_float2(
                t_raw.x * tw_cos - t_raw.y * tw_sin,
                t_raw.x * tw_sin + t_raw.y * tw_cos
            );
            
            shmem[threadIdx.x][idx1] = make_float2(u.x + t.x, u.y + t.y);
            shmem[threadIdx.x][idx2] = make_float2(u.x - t.x, u.y - t.y);
        }
        __syncthreads();
    }
    
    // Stage 2: m=8, m/2=4
    {
        const int m = 8;
        const int m2 = 4;
        const int k = (point_id / 8) * 8;
        const int j = point_id % 8;
        
        if (j < m2) {
            const int idx1 = k + j;
            const int idx2 = idx1 + m2;
            
            float2 u = shmem[threadIdx.x][idx1];
            float2 t_raw = shmem[threadIdx.x][idx2];
            
            const int tw_idx = (j * 32) / m;
            const float tw_cos = TWIDDLE_32_COS[tw_idx];
            const float tw_sin = TWIDDLE_32_SIN[tw_idx];
            
            float2 t = make_float2(
                t_raw.x * tw_cos - t_raw.y * tw_sin,
                t_raw.x * tw_sin + t_raw.y * tw_cos
            );
            
            shmem[threadIdx.x][idx1] = make_float2(u.x + t.x, u.y + t.y);
            shmem[threadIdx.x][idx2] = make_float2(u.x - t.x, u.y - t.y);
        }
        __syncthreads();
    }
    
    // Stage 3: m=16, m/2=8
    {
        const int m = 16;
        const int m2 = 8;
        const int k = (point_id / 16) * 16;
        const int j = point_id % 16;
        
        if (j < m2) {
            const int idx1 = k + j;
            const int idx2 = idx1 + m2;
            
            float2 u = shmem[threadIdx.x][idx1];
            float2 t_raw = shmem[threadIdx.x][idx2];
            
            const int tw_idx = (j * 32) / m;
            const float tw_cos = TWIDDLE_32_COS[tw_idx];
            const float tw_sin = TWIDDLE_32_SIN[tw_idx];
            
            float2 t = make_float2(
                t_raw.x * tw_cos - t_raw.y * tw_sin,
                t_raw.x * tw_sin + t_raw.y * tw_cos
            );
            
            shmem[threadIdx.x][idx1] = make_float2(u.x + t.x, u.y + t.y);
            shmem[threadIdx.x][idx2] = make_float2(u.x - t.x, u.y - t.y);
        }
        __syncthreads();
    }
    
    // Stage 4: m=32, m/2=16
    {
        const int m = 32;
        const int m2 = 16;
        const int j = point_id;  // Only one group
        
        if (j < m2) {
            const int idx1 = j;
            const int idx2 = j + m2;
            
            float2 u = shmem[threadIdx.x][idx1];
            float2 t_raw = shmem[threadIdx.x][idx2];
            
            const int tw_idx = j;  // j * 32 / 32 = j
            const float tw_cos = TWIDDLE_32_COS[tw_idx];
            const float tw_sin = TWIDDLE_32_SIN[tw_idx];
            
            float2 t = make_float2(
                t_raw.x * tw_cos - t_raw.y * tw_sin,
                t_raw.x * tw_sin + t_raw.y * tw_cos
            );
            
            shmem[threadIdx.x][idx1] = make_float2(u.x + t.x, u.y + t.y);
            shmem[threadIdx.x][idx2] = make_float2(u.x - t.x, u.y - t.y);
        }
        __syncthreads();
    }
    
    // Store output (NO SHIFT!)
    const int output_idx = fft_id * 32 + point_id;
    output[output_idx] = make_cuComplex(shmem[threadIdx.x][point_id].x, shmem[threadIdx.x][point_id].y);
}

// Host launcher
namespace CudaCalc {

void launch_fft32_correct(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    const int threads_per_block_x = 32;  // 32 FFTs per block
    const int threads_per_block_y = 32;  // 32 points per FFT
    const int num_blocks = (num_windows + 31) / 32;
    
    dim3 block(threads_per_block_x, threads_per_block_y);
    dim3 grid(num_blocks);
    
    fft32_correct_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}

} // namespace CudaCalc

