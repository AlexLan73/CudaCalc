/**
 * @file fft16_wmma.cpp
 * @brief FFT16_WMMA wrapper implementation
 */

#include "ModelsFunction/include/nvidia/fft/fft16_wmma.h"
#include "Interface/include/common_types.h"
#include <iostream>
#include <stdexcept>

namespace CudaCalc {

// Forward declaration of kernel launcher
void launch_fft16_wmma(
    const cuComplex* d_input,
    cuComplex* d_output,
    int num_windows,
    cudaStream_t stream
);

FFT16_WMMA::FFT16_WMMA() {
    // Constructor
}

FFT16_WMMA::~FFT16_WMMA() {
    cleanup();
}

bool FFT16_WMMA::initialize() {
    if (initialized_) {
        std::cerr << "Warning: FFT16_WMMA already initialized" << std::endl;
        return true;
    }
    
    // Check for Tensor Core support
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    if (prop.major < 7) {
        throw std::runtime_error("FFT16_WMMA requires compute capability >= 7.0 (Volta or newer)");
    }
    
    initialized_ = true;
    
    std::cout << "FFT16_WMMA initialized (Tensor Core support: compute " 
              << prop.major << "." << prop.minor << ")" << std::endl;
    return true;
}

void FFT16_WMMA::cleanup() {
    if (d_input_) {
        CUDA_CHECK(cudaFree(d_input_));
        d_input_ = nullptr;
    }
    
    if (d_output_) {
        CUDA_CHECK(cudaFree(d_output_));
        d_output_ = nullptr;
    }
    
    initialized_ = false;
}

OutputSpectralData FFT16_WMMA::process(const InputSignalData& input) {
    if (!initialized_) {
        throw std::runtime_error("FFT16_WMMA not initialized! Call initialize() first.");
    }
    
    // Validate input
    if (!input.is_valid()) {
        throw std::invalid_argument("Invalid input signal data");
    }
    
    if (input.config.window_fft != 16) {
        throw std::invalid_argument("FFT16_WMMA requires window_fft = 16");
    }
    
    const int total_points = input.signal.size();
    num_windows_ = total_points / 16;
    
    std::cout << "\nFFT16_WMMA::process()" << std::endl;
    std::cout << "  Total points: " << total_points << std::endl;
    std::cout << "  Windows: " << num_windows_ << std::endl;
    
    // === ALLOCATE DEVICE MEMORY ===
    const size_t data_size = total_points * sizeof(cuComplex);
    
    CUDA_CHECK(cudaMalloc(&d_input_, data_size));
    CUDA_CHECK(cudaMalloc(&d_output_, data_size));
    
    // === UPLOAD: Host → Device ===
    CUDA_CHECK(cudaMemcpy(
        d_input_,
        reinterpret_cast<const cuComplex*>(input.signal.data()),
        data_size,
        cudaMemcpyHostToDevice
    ));
    
    // === COMPUTE: Execute FFT kernel ===
    launch_fft16_wmma(d_input_, d_output_, num_windows_, 0);
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // === DOWNLOAD: Device → Host ===
    std::vector<cuComplex> output_data(total_points);
    CUDA_CHECK(cudaMemcpy(
        output_data.data(),
        d_output_,
        data_size,
        cudaMemcpyDeviceToHost
    ));
    
    // === ORGANIZE INTO WINDOWS ===
    OutputSpectralData output;
    output.windows.resize(num_windows_);
    
    for (int w = 0; w < num_windows_; ++w) {
        output.windows[w].resize(16);
        for (int p = 0; p < 16; ++p) {
            const cuComplex& val = output_data[w * 16 + p];
            output.windows[w][p] = std::complex<float>(val.x, val.y);
        }
    }
    
    std::cout << "  FFT16_WMMA complete ✓" << std::endl;
    
    return output;
}

} // namespace CudaCalc

