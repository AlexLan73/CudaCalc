/**
 * @file fft32_3D_block.cu
 * @brief FFT32 with 3D block structure
 * 
 * Block: [64 FFT, 8 points, 2 layers] = 1024 threads
 * Each thread in layer processes multiple points
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

static __device__ int bitReverse5_3d(int x) {
    int result = 0;
    result |= (x & 1) << 4;
    result |= (x & 2) << 2;
    result |= (x & 4);
    result |= (x & 8) >> 2;
    result |= (x & 16) >> 4;
    return result;
}

/**
 * FFT32 with 3D block structure
 * Block: [64, 8, 2] = 1024 threads
 * 
 * X: 0-63 (which FFT)
 * Y: 0-7 (base point index)
 * Z: 0-1 (layer: processes Y+0 and Y+16 OR different point sets)
 */
__global__ void fft32_3D_kernel(
    const cuComplex* __restrict__ input,
    cuComplex* __restrict__ output,
    int num_windows
) {
    const int x = threadIdx.x;  // 0-63: which FFT
    const int y = threadIdx.y;  // 0-7: base index
    const int z = threadIdx.z;  // 0-1: layer
    const int global_fft_id = blockIdx.x * 64 + x;
    
    if (global_fft_id >= num_windows) return;
    
    // Shared memory: [64 FFT][32 points] + padding
    __shared__ float2 shmem[64][34];
    
    // Each thread loads 2 points (based on layer)
    // Layer 0: points 0-15, Layer 1: points 16-31
    int point_idx1 = y + z * 16;          // First point
    int point_idx2 = y + 8 + z * 16;      // Second point (y+8)
    
    // Load first point
    {
        const int input_idx = global_fft_id * 32 + point_idx1;
        const int reversed_idx = bitReverse5_3d(point_idx1);
        const cuComplex val = __ldg(&input[input_idx]);
        shmem[x][reversed_idx] = make_float2(val.x, val.y);
    }
    
    // Load second point
    {
        const int input_idx = global_fft_id * 32 + point_idx2;
        const int reversed_idx = bitReverse5_3d(point_idx2);
        const cuComplex val = __ldg(&input[input_idx]);
        shmem[x][reversed_idx] = make_float2(val.x, val.y);
    }
    __syncthreads();
    
    // === BUTTERFLY STAGES ===
    // Only Y-dimension threads (0-7) work, Z just helps with loading
    if (z == 0 && y < 8) {
        // STAGE 0: m=2, m2=1
        // Need to process 16 pairs, we have 8 threads
        // Each thread processes 2 pairs
        for (int p = 0; p < 2; ++p) {
            int tid_virtual = y + p * 8;  // 0-15
            const int m2 = 1;
            const int k = (tid_virtual / m2) * 2;
            const int j = tid_virtual % m2;
            const int idx1 = k + j;
            const int idx2 = idx1 + m2;
            
            const int twiddle_idx = j * 16;
            const float tw_cos = TWIDDLE_32_COS[twiddle_idx & 15];
            const float tw_sin = TWIDDLE_32_SIN[twiddle_idx & 15];
            
            float2 u = shmem[x][idx1];
            float2 v = shmem[x][idx2];
            
            float2 t;
            t.x = v.x * tw_cos - v.y * tw_sin;
            t.y = v.x * tw_sin + v.y * tw_cos;
            
            shmem[x][idx1] = make_float2(u.x + t.x, u.y + t.y);
            shmem[x][idx2] = make_float2(u.x - t.x, u.y - t.y);
        }
    }
    __syncthreads();
    
    // Similar for other stages...
    // (For now, just store to test loading)
    
    // Store - each thread stores 2 points
    {
        const int output_idx1 = global_fft_id * 32 + point_idx1;
        output[output_idx1] = make_cuComplex(shmem[x][point_idx1].x, shmem[x][point_idx1].y);
    }
    {
        const int output_idx2 = global_fft_id * 32 + point_idx2;
        output[output_idx2] = make_cuComplex(shmem[x][point_idx2].x, shmem[x][point_idx2].y);
    }
}

extern "C" void launch_fft32_3D(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
) {
    dim3 block(64, 8, 2);  // [64 FFT, 8 points, 2 layers] = 1024
    int num_blocks = (num_windows + 63) / 64;
    dim3 grid(num_blocks);
    
    fft32_3D_kernel<<<grid, block>>>(d_input, d_output, num_windows);
}



