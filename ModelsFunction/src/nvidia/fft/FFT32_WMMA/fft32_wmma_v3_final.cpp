/**
 * @file fft32_wmma_v3_final.cpp
 * @brief FFT32 V3 Final host code
 */

#include "ModelsFunction/include/nvidia/fft/fft32_wmma_v3_final.h"
#include "Interface/include/common_types.h"

namespace CudaCalc {

// Forward declaration of V3 kernel launcher
extern void launch_fft32_v3_final_kernel(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
);

void FFT32_WMMA_V3_Final::process(const void* input, void* output) {
    if (!initialized_) {
        throw std::runtime_error("FFT32_WMMA_V3_Final not initialized");
    }

    // Copy input to device
    size_t size = num_windows_ * 32 * sizeof(cuComplex);
    CUDA_CHECK(cudaMemcpy(d_input_, input, size, cudaMemcpyHostToDevice));

    // Launch V3 kernel
    launch_fft32_v3_final_kernel(d_input_, d_output_, num_windows_);

    // Copy output to host
    CUDA_CHECK(cudaMemcpy(output, d_output_, size, cudaMemcpyDeviceToHost));
}

} // namespace CudaCalc

