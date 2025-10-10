/**
 * @file fft16_wmma_optimized_profiled.cpp
 */

#include "ModelsFunction/include/nvidia/fft/fft16_wmma_optimized_profiled.h"
#include "Interface/include/common_types.h"

namespace CudaCalc {

// Forward declaration
void launch_fft16_wmma_optimized(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows,
    cudaStream_t stream
);

OutputSpectralData FFT16_WMMA_Optimized_Profiled::process_with_profiling(
    const InputSignalData& input,
    BasicProfilingResult& profiling
) {
    if (!initialized_) {
        throw std::runtime_error("Not initialized!");
    }
    
    if (!input.is_valid() || input.config.window_fft != 16) {
        throw std::invalid_argument("Invalid input");
    }
    
    const int total_points = input.signal.size();
    num_windows_ = total_points / 16;
    const size_t data_size = total_points * sizeof(cuComplex);
    
    // Allocate
    CUDA_CHECK(cudaMalloc(&d_input_, data_size));
    CUDA_CHECK(cudaMalloc(&d_output_, data_size));
    
    // === UPLOAD with profiling ===
    profiler_.start_upload();
    CUDA_CHECK(cudaMemcpy(d_input_, reinterpret_cast<const cuComplex*>(input.signal.data()),
                          data_size, cudaMemcpyHostToDevice));
    profiler_.end_upload();
    
    // === COMPUTE with profiling ===
    profiler_.start_compute();
    launch_fft16_wmma_optimized(d_input_, d_output_, num_windows_, 0);
    CUDA_CHECK(cudaGetLastError());
    profiler_.end_compute();
    
    // === DOWNLOAD with profiling ===
    std::vector<cuComplex> output_data(total_points);
    profiler_.start_download();
    CUDA_CHECK(cudaMemcpy(output_data.data(), d_output_, data_size, cudaMemcpyDeviceToHost));
    profiler_.end_download();
    
    // Get profiling
    profiling = profiler_.get_results(get_name(), input.config);
    
    // Organize
    OutputSpectralData output;
    output.windows.resize(num_windows_);
    
    for (int w = 0; w < num_windows_; ++w) {
        output.windows[w].resize(16);
        for (int p = 0; p < 16; ++p) {
            const cuComplex& val = output_data[w * 16 + p];
            output.windows[w][p] = std::complex<float>(val.x, val.y);
        }
    }
    
    return output;
}

} // namespace CudaCalc

