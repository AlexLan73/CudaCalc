/**
 * @file fft16_wmma_ultra.cpp
 * @brief FFT16_WMMA_Ultra wrapper implementation
 */

#include "ModelsFunction/include/nvidia/fft/fft16_wmma_ultra.h"
#include "Interface/include/common_types.h"
#include <iostream>
#include <vector>

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

FFT16_WMMA_Ultra::FFT16_WMMA_Ultra() {
}

FFT16_WMMA_Ultra::~FFT16_WMMA_Ultra() {
    cleanup();
}

bool FFT16_WMMA_Ultra::initialize() {
    if (initialized_) {
        std::cerr << "Warning: FFT16_WMMA_Ultra already initialized" << std::endl;
        return true;
    }
    
    // Check Tensor Core support
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    if (prop.major < 7) {
        throw std::runtime_error("FFT16_WMMA_Ultra requires compute capability >= 7.0");
    }
    
    initialized_ = true;
    
    std::cout << "FFT16_WMMA_Ultra initialized (REAL Tensor Cores FP16, compute " 
              << prop.major << "." << prop.minor << ")" << std::endl;
    return true;
}

void FFT16_WMMA_Ultra::cleanup() {
    if (d_input_real_) CUDA_CHECK(cudaFree(d_input_real_));
    if (d_input_imag_) CUDA_CHECK(cudaFree(d_input_imag_));
    if (d_output_real_) CUDA_CHECK(cudaFree(d_output_real_));
    if (d_output_imag_) CUDA_CHECK(cudaFree(d_output_imag_));
    
    d_input_real_ = nullptr;
    d_input_imag_ = nullptr;
    d_output_real_ = nullptr;
    d_output_imag_ = nullptr;
    
    initialized_ = false;
}

OutputSpectralData FFT16_WMMA_Ultra::process(const InputSignalData& input) {
    if (!initialized_) {
        throw std::runtime_error("FFT16_WMMA_Ultra not initialized!");
    }
    
    if (!input.is_valid() || input.config.window_fft != 16) {
        throw std::invalid_argument("Invalid input or wrong window size");
    }
    
    const int total_points = input.signal.size();
    num_windows_ = total_points / 16;
    
    std::cout << "\nFFT16_WMMA_Ultra::process() [FP16 Tensor Cores]" << std::endl;
    std::cout << "  Windows: " << num_windows_ << std::endl;
    
    // === ALLOCATE DEVICE MEMORY (FP16, separate real/imag!) ===
    const size_t fp16_size = total_points * sizeof(__half);
    
    CUDA_CHECK(cudaMalloc(&d_input_real_, fp16_size));
    CUDA_CHECK(cudaMalloc(&d_input_imag_, fp16_size));
    CUDA_CHECK(cudaMalloc(&d_output_real_, fp16_size));
    CUDA_CHECK(cudaMalloc(&d_output_imag_, fp16_size));
    
    // === CONVERT FP32 → FP16 & SEPARATE real/imag ===
    std::vector<__half> h_input_real(total_points);
    std::vector<__half> h_input_imag(total_points);
    
    for (int i = 0; i < total_points; ++i) {
        h_input_real[i] = __float2half(input.signal[i].real());
        h_input_imag[i] = __float2half(input.signal[i].imag());
    }
    
    // === UPLOAD: Host → Device (FP16!) ===
    CUDA_CHECK(cudaMemcpy(d_input_real_, h_input_real.data(), fp16_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_imag_, h_input_imag.data(), fp16_size, cudaMemcpyHostToDevice));
    
    // === COMPUTE: ULTRA FFT kernel ===
    launch_fft16_wmma_ultra(d_input_real_, d_input_imag_, 
                            d_output_real_, d_output_imag_,
                            num_windows_, 0);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // === DOWNLOAD: Device → Host (FP16) ===
    std::vector<__half> h_output_real(total_points);
    std::vector<__half> h_output_imag(total_points);
    
    CUDA_CHECK(cudaMemcpy(h_output_real.data(), d_output_real_, fp16_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_imag.data(), d_output_imag_, fp16_size, cudaMemcpyDeviceToHost));
    
    // === CONVERT FP16 → FP32 & ORGANIZE ===
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
    
    std::cout << "  FFT16_WMMA_Ultra complete ✓" << std::endl;
    
    return output;
}

} // namespace CudaCalc

