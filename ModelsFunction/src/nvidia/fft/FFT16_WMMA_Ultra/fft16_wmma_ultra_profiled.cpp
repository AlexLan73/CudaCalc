/**
 * @file fft16_wmma_ultra_profiled.cpp
 * @brief FFT16_WMMA_Ultra with profiling implementation
 */

#include "ModelsFunction/include/nvidia/fft/fft16_wmma_ultra_profiled.h"
#include "Interface/include/common_types.h"

namespace CudaCalc {

// Forward declaration
void launch_fft16_wmma_ultra(
    const __half* d_input_real,
    const __half* d_input_imag,
    __half* d_output_real,
    __half* d_output_imag,
    int num_ffts,
    cudaStream_t stream
);

OutputSpectralData FFT16_WMMA_Ultra_Profiled::process_with_profiling(
    const InputSignalData& input,
    BasicProfilingResult& profiling
) {
    if (!initialized_) {
        throw std::runtime_error("FFT16_WMMA_Ultra_Profiled not initialized!");
    }
    
    if (!input.is_valid() || input.config.window_fft != 16) {
        throw std::invalid_argument("Invalid input");
    }
    
    const int total_points = input.signal.size();
    num_windows_ = total_points / 16;
    const size_t fp16_size = total_points * sizeof(__half);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input_real_, fp16_size));
    CUDA_CHECK(cudaMalloc(&d_input_imag_, fp16_size));
    CUDA_CHECK(cudaMalloc(&d_output_real_, fp16_size));
    CUDA_CHECK(cudaMalloc(&d_output_imag_, fp16_size));
    
    // Convert FP32 → FP16 & separate
    std::vector<__half> h_input_real(total_points);
    std::vector<__half> h_input_imag(total_points);
    
    for (int i = 0; i < total_points; ++i) {
        h_input_real[i] = __float2half(input.signal[i].real());
        h_input_imag[i] = __float2half(input.signal[i].imag());
    }
    
    // === UPLOAD with profiling ===
    profiler_.start_upload();
    CUDA_CHECK(cudaMemcpy(d_input_real_, h_input_real.data(), fp16_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_imag_, h_input_imag.data(), fp16_size, cudaMemcpyHostToDevice));
    profiler_.end_upload();
    
    // === COMPUTE with profiling ===
    profiler_.start_compute();
    launch_fft16_wmma_ultra(d_input_real_, d_input_imag_,
                            d_output_real_, d_output_imag_,
                            num_windows_, 0);
    CUDA_CHECK(cudaGetLastError());
    profiler_.end_compute();
    
    // === DOWNLOAD with profiling ===
    std::vector<__half> h_output_real(total_points);
    std::vector<__half> h_output_imag(total_points);
    
    profiler_.start_download();
    CUDA_CHECK(cudaMemcpy(h_output_real.data(), d_output_real_, fp16_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_imag.data(), d_output_imag_, fp16_size, cudaMemcpyDeviceToHost));
    profiler_.end_download();
    
    // Get profiling results
    profiling = profiler_.get_results(get_name(), input.config);
    
    // Convert FP16 → FP32 & organize
    OutputSpectralData output;
    output.windows.resize(num_windows_);
    
    for (int w = 0; w < num_windows_; ++w) {
        output.windows[w].resize(16);
        for (int p = 0; p < 16; ++p) {
            int idx = w * 16 + p;
            float real_val = __half2float(h_output_real[idx]);
            float imag_val = __half2float(h_output_imag[idx]);
            output.windows[w][p] = std::complex<float>(real_val, imag_val);
        }
    }
    
    return output;
}

} // namespace CudaCalc

