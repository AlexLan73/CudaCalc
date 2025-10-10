/**
 * @file fft32_wmma_optimized.cpp
 * @brief FFT32_WMMA_Optimized class implementation (host code)
 */

#include "ModelsFunction/include/nvidia/fft/fft32_wmma_optimized.h"
#include "Interface/include/common_types.h"
#include <stdexcept>

// Forward declaration of CUDA kernel
extern void launch_fft32_wmma_optimized_kernel(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
);

using namespace CudaCalc;

FFT32_WMMA_Optimized::FFT32_WMMA_Optimized()
    : d_input_(nullptr), d_output_(nullptr), num_windows_(0), initialized_(false) {}

FFT32_WMMA_Optimized::~FFT32_WMMA_Optimized() {
    cleanup();
}

void FFT32_WMMA_Optimized::initialize(int num_windows) {
    if (initialized_) {
        cleanup();
    }

    num_windows_ = num_windows;

    // Allocate device memory
    size_t size = num_windows * 32 * sizeof(cuComplex);
    CUDA_CHECK(cudaMalloc(&d_input_, size));
    CUDA_CHECK(cudaMalloc(&d_output_, size));

    initialized_ = true;
}

void FFT32_WMMA_Optimized::process(const void* input, void* output) {
    if (!initialized_) {
        throw std::runtime_error("FFT32_WMMA_Optimized not initialized");
    }

    // Copy input to device
    size_t size = num_windows_ * 32 * sizeof(cuComplex);
    CUDA_CHECK(cudaMemcpy(d_input_, input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    launch_fft32_wmma_optimized_kernel(d_input_, d_output_, num_windows_);

    // Copy output to host
    CUDA_CHECK(cudaMemcpy(output, d_output_, size, cudaMemcpyDeviceToHost));
}

void FFT32_WMMA_Optimized::cleanup() {
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

