/**
 * @file fft32_wmma_optimized_profiled.cpp
 * @brief FFT32 profiled version implementation
 */

#include "ModelsFunction/include/nvidia/fft/fft32_wmma_optimized_profiled.h"
#include "Tester/include/performance/basic_profiler.h"
#include "Interface/include/common_types.h"

// Forward declaration of CUDA kernel
extern void launch_fft32_wmma_optimized_kernel(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows
);

using namespace CudaCalc;

void FFT32_WMMA_Optimized_Profiled::process_with_profiling(
    const void* input, void* output, BasicProfiler& profiler
) {
    if (!initialized_) {
        throw std::runtime_error("FFT32_WMMA_Optimized_Profiled not initialized");
    }

    size_t size = num_windows_ * 32 * sizeof(cuComplex);

    // === Upload ===
    profiler.start_upload();
    CUDA_CHECK(cudaMemcpy(d_input_, input, size, cudaMemcpyHostToDevice));
    profiler.end_upload();

    // === Compute ===
    profiler.start_compute();
    launch_fft32_wmma_optimized_kernel(d_input_, d_output_, num_windows_);
    profiler.end_compute();

    // === Download ===
    profiler.start_download();
    CUDA_CHECK(cudaMemcpy(output, d_output_, size, cudaMemcpyDeviceToHost));
    profiler.end_download();
}

