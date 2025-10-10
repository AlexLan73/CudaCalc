/**
 * @file fft64_wmma_optimized.cpp
 */

#include "ModelsFunction/include/nvidia/fft/fft64_wmma_optimized.h"
#include "Interface/include/common_types.h"

namespace CudaCalc {

extern void launch_fft64_optimized(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
);

void FFT64_WMMA_Optimized::initialize(int num_windows) {
    num_windows_ = num_windows;
    size_t size = num_windows * 64 * sizeof(cuComplex);
    
    CUDA_CHECK(cudaMalloc(&d_input_, size));
    CUDA_CHECK(cudaMalloc(&d_output_, size));
    
    initialized_ = true;
}

void FFT64_WMMA_Optimized::process(const void* input, void* output) {
    if (!initialized_) {
        throw std::runtime_error("FFT64_WMMA_Optimized not initialized");
    }

    size_t size = num_windows_ * 64 * sizeof(cuComplex);
    CUDA_CHECK(cudaMemcpy(d_input_, input, size, cudaMemcpyHostToDevice));
    
    launch_fft64_optimized(d_input_, d_output_, num_windows_);
    
    CUDA_CHECK(cudaMemcpy(output, d_output_, size, cudaMemcpyDeviceToHost));
}

void FFT64_WMMA_Optimized::cleanup() {
    if (d_input_) {
        cudaFree(d_input_);
        d_input_ = nullptr;
    }
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
    initialized_ = false;
}

} // namespace CudaCalc

