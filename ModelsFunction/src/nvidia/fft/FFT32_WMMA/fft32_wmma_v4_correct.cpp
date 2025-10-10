/**
 * @file fft32_wmma_v4_correct.cpp
 */

#include "ModelsFunction/include/nvidia/fft/fft32_wmma_v4_correct.h"
#include "Interface/include/common_types.h"

namespace CudaCalc {

extern void launch_fft32_v4_correct(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows,
    cudaStream_t stream
);

void FFT32_WMMA_V4_Correct::process(const void* input, void* output) {
    if (!initialized_) {
        throw std::runtime_error("FFT32_WMMA_V4_Correct not initialized");
    }

    size_t size = num_windows_ * 32 * sizeof(cuComplex);
    CUDA_CHECK(cudaMemcpy(d_input_, input, size, cudaMemcpyHostToDevice));

    launch_fft32_v4_correct(d_input_, d_output_, num_windows_, 0);

    CUDA_CHECK(cudaMemcpy(output, d_output_, size, cudaMemcpyDeviceToHost));
}

} // namespace CudaCalc

