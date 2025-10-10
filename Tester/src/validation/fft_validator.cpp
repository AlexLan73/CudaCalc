/**
 * @file fft_validator.cpp
 * @brief FFTValidator implementation
 */

#include "Tester/include/validation/fft_validator.h"
#include "Interface/include/common_types.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace CudaCalc {

FFTValidator::FFTValidator(double tolerance)
    : tolerance_(tolerance)
{
    // Delay plan creation until validation
}

FFTValidator::~FFTValidator() {
    if (plan_initialized_) {
        cufftDestroy(cufft_plan_);
    }
}

ValidationResult FFTValidator::validate(
    const InputSignalData& input,
    const OutputSpectralData& output,
    const std::string& algorithm
) {
    ValidationResult result;
    result.algorithm = algorithm;
    result.reference = "cuFFT";
    result.tolerance = tolerance_;
    result.total_points = input.signal.size();
    result.failed_points = 0;
    
    const int window_size = input.config.window_fft;
    const int num_windows = input.config.num_windows();
    
    std::cout << "\nFFTValidator::validate()" << std::endl;
    std::cout << "  Algorithm: " << algorithm << std::endl;
    std::cout << "  Tolerance: " << (tolerance_ * 100.0) << "%" << std::endl;
    std::cout << "  Windows: " << num_windows << " × " << window_size << " points" << std::endl;
    
    // === ALLOCATE DEVICE MEMORY FOR cuFFT ===
    cufftComplex* d_input_ref;
    cufftComplex* d_output_ref;
    const size_t total_size = result.total_points * sizeof(cufftComplex);
    
    CUDA_CHECK(cudaMalloc(&d_input_ref, total_size));
    CUDA_CHECK(cudaMalloc(&d_output_ref, total_size));
    
    // === UPLOAD INPUT ===
    CUDA_CHECK(cudaMemcpy(
        d_input_ref,
        reinterpret_cast<const cufftComplex*>(input.signal.data()),
        total_size,
        cudaMemcpyHostToDevice
    ));
    
    // === CREATE cuFFT PLAN ===
    // Batch FFT: multiple FFTs of size window_size
    int rank = 1;                    // 1D FFT
    int n[1] = {window_size};        // FFT size
    int batch = num_windows;         // Number of FFTs
    
    CUFFT_CHECK(cufftPlanMany(
        &cufft_plan_,
        rank, n,
        nullptr, 1, window_size,     // inembed, istride, idist
        nullptr, 1, window_size,     // onembed, ostride, odist
        CUFFT_C2C,                   // Complex-to-Complex
        batch                        // Number of transforms
    ));
    
    plan_initialized_ = true;
    
    // === EXECUTE cuFFT (REFERENCE) ===
    CUFFT_CHECK(cufftExecC2C(
        cufft_plan_,
        d_input_ref,
        d_output_ref,
        CUFFT_FORWARD  // Forward FFT
    ));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // === DOWNLOAD REFERENCE RESULTS ===
    std::vector<cufftComplex> reference_output(result.total_points);
    CUDA_CHECK(cudaMemcpy(
        reference_output.data(),
        d_output_ref,
        total_size,
        cudaMemcpyDeviceToHost
    ));
    
    // === COMPARE RESULTS ===
    double sum_error = 0.0;
    result.max_relative_error = 0.0;
    
    for (int w = 0; w < num_windows; ++w) {
        for (int p = 0; p < window_size; ++p) {
            // cuFFT output (no FFT shift) - need to apply shift to match our output
            int ref_idx = w * window_size + p;
            
            // Apply FFT shift to reference to match our shifted output
            // cuFFT: [DC, 1, ..., 7, 8, -7, ..., -1]
            // Ours:  [-8, -7, ..., -1, DC, 1, ..., 7]
            int shifted_p = p;
            if (p < window_size / 2) {
                shifted_p = p + window_size / 2;  // Lower half of cuFFT → upper half of ours
            } else {
                shifted_p = p - window_size / 2;  // Upper half of cuFFT → lower half of ours
            }
            
            const cufftComplex& ref = reference_output[ref_idx];
            const std::complex<float>& our = output.windows[w][shifted_p];
            
            // Calculate error
            float ref_real = ref.x;
            float ref_imag = ref.y;
            float our_real = our.real();
            float our_imag = our.imag();
            
            float ref_mag = std::sqrt(ref_real*ref_real + ref_imag*ref_imag);
            float error_real = std::abs(ref_real - our_real);
            float error_imag = std::abs(ref_imag - our_imag);
            float error_mag = std::sqrt(error_real*error_real + error_imag*error_imag);
            
            double relative_error = 0.0;
            if (ref_mag > 1e-10) {  // Avoid division by zero
                relative_error = error_mag / ref_mag;
            } else {
                relative_error = error_mag;  // Absolute error for near-zero values
            }
            
            sum_error += relative_error;
            result.max_relative_error = std::max(result.max_relative_error, relative_error);
            
            if (relative_error > tolerance_) {
                result.failed_points++;
            }
        }
    }
    
    result.avg_relative_error = sum_error / result.total_points;
    result.passed = (result.failed_points == 0);
    
    // === CLEANUP ===
    CUDA_CHECK(cudaFree(d_input_ref));
    CUDA_CHECK(cudaFree(d_output_ref));
    
    // === PRINT RESULTS ===
    std::cout << std::endl;
    std::cout << "  Validation Result:" << std::endl;
    std::cout << "    Max relative error: " << (result.max_relative_error * 100.0) << "%" << std::endl;
    std::cout << "    Avg relative error: " << (result.avg_relative_error * 100.0) << "%" << std::endl;
    std::cout << "    Failed points: " << result.failed_points << " / " << result.total_points << std::endl;
    std::cout << "    Status: " << (result.passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    std::cout << std::endl;
    
    return result;
}

} // namespace CudaCalc

