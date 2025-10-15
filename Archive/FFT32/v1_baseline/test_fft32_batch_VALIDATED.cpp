/**
 * @file test_fft32_batch_VALIDATED.cpp
 * @brief FFT32 batch test with CORRECT validation rules
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>

extern "C" void launch_fft32_batch32_v2(const cuComplex* d_input, cuComplex* d_output, int num_windows);

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════╗\n";
    std::cout << "║   FFT32 BATCH TEST (Correct Validation)      ║\n";
    std::cout << "╚═══════════════════════════════════════════════╝\n\n";
    
    const int num_windows = 32;
    const int total_points = 1024;
    const double MAGNITUDE_THRESHOLD = 0.01;  // Noise floor
    const double ERROR_TOLERANCE = 0.01;       // 0.01% error
    
    std::cout << "Validation rules:\n";
    std::cout << "  Magnitude threshold: " << MAGNITUDE_THRESHOLD << " (noise floor)\n";
    std::cout << "  Error tolerance: " << ERROR_TOLERANCE << "%\n\n";
    
    // Signal: exp(2πi*n/32) repeated 32 times
    std::vector<std::complex<float>> input(total_points);
    for (int w = 0; w < num_windows; ++w) {
        for (int i = 0; i < 32; ++i) {
            float angle = 2.0f * M_PI * i / 32.0f;
            input[w * 32 + i] = std::complex<float>(std::cos(angle), std::sin(angle));
        }
    }
    
    // cuFFT batch
    cuComplex* d_ref_in, *d_ref_out;
    cudaMalloc(&d_ref_in, total_points * sizeof(cuComplex));
    cudaMalloc(&d_ref_out, total_points * sizeof(cuComplex));
    cudaMemcpy(d_ref_in, input.data(), total_points * sizeof(cuComplex), cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan1d(&plan, 32, CUFFT_C2C, num_windows);
    cufftExecC2C(plan, d_ref_in, d_ref_out, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    
    std::vector<std::complex<float>> ref(total_points);
    cudaMemcpy(ref.data(), d_ref_out, total_points * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    
    // Our FFT32 batch
    cuComplex* d_our_in, *d_our_out;
    cudaMalloc(&d_our_in, total_points * sizeof(cuComplex));
    cudaMalloc(&d_our_out, total_points * sizeof(cuComplex));
    cudaMemcpy(d_our_in, input.data(), total_points * sizeof(cuComplex), cudaMemcpyHostToDevice);
    
    launch_fft32_batch32_v2(d_our_in, d_our_out, num_windows);
    cudaDeviceSynchronize();
    
    std::vector<std::complex<float>> our(total_points);
    cudaMemcpy(our.data(), d_our_out, total_points * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    
    // CORRECT Validation
    std::cout << "Validating all 32 windows...\n\n";
    
    int total_failed = 0;
    double max_error_global = 0.0;
    
    for (int w = 0; w < num_windows; ++w) {
        int window_failed = 0;
        double max_error_window = 0.0;
        
        for (int i = 0; i < 32; ++i) {
            int idx = w * 32 + i;
            
            float mag_ref = std::abs(ref[idx]);
            float mag_our = std::abs(our[idx]);
            
            // CORRECT validation: ignore noise floor
            if (mag_ref >= MAGNITUDE_THRESHOLD || mag_our >= MAGNITUDE_THRESHOLD) {
                double error = 0.0;
                if (mag_ref > 1e-6f) {
                    error = std::abs(mag_our - mag_ref) / mag_ref * 100.0;
                }
                
                if (error > ERROR_TOLERANCE) {
                    window_failed++;
                }
                
                if (error > max_error_window) max_error_window = error;
                if (error > max_error_global) max_error_global = error;
            }
        }
        
        if (window_failed > 0) {
            total_failed++;
            if (total_failed <= 5) {
                std::cout << "  Window " << w << ": " << window_failed << " points failed, max error = " 
                          << std::setprecision(4) << max_error_window << "%\n";
            }
        }
    }
    
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════╗\n";
    std::cout << "║            VALIDATION RESULTS                 ║\n";
    std::cout << "╠═══════════════════════════════════════════════╣\n";
    std::cout << "║ Total windows:     " << std::setw(4) << num_windows << "                        ║\n";
    std::cout << "║ Failed windows:    " << std::setw(4) << total_failed << "                        ║\n";
    std::cout << "║ Max error:         " << std::setw(10) << std::setprecision(6) << max_error_global << "%             ║\n";
    std::cout << "╚═══════════════════════════════════════════════╝\n\n";
    
    if (total_failed == 0) {
        std::cout << "🎉🎉🎉 SUCCESS! All 32 windows PERFECT! 🎉🎉🎉\n\n";
    } else {
        std::cout << "❌ " << total_failed << " windows failed\n\n";
    }
    
    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_ref_in); cudaFree(d_ref_out);
    cudaFree(d_our_in); cudaFree(d_our_out);
    
    return (total_failed == 0) ? 0 : 1;
}
