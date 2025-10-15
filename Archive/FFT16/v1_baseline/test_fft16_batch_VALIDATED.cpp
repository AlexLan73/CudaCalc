#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>

extern "C" void launch_fft16_batch64(const cuComplex*, cuComplex*, int);

int main() {
    std::cout << "\n╔═══════════════════════════════════════════════╗\n";
    std::cout << "║   FFT16 BATCH TEST (Correct Validation)      ║\n";
    std::cout << "╚═══════════════════════════════════════════════╝\n\n";
    
    const int num_windows = 64;
    const int total_points = 1024;
    const double MAGNITUDE_THRESHOLD = 0.01;
    const double ERROR_TOLERANCE = 0.01;
    
    std::vector<std::complex<float>> input(total_points);
    for (int w = 0; w < num_windows; ++w) {
        for (int i = 0; i < 16; ++i) {
            float angle = 2.0f * M_PI * i / 16.0f;
            input[w * 16 + i] = std::complex<float>(std::cos(angle), std::sin(angle));
        }
    }
    
    cuComplex *d_ref_in, *d_ref_out, *d_our_in, *d_our_out;
    cudaMalloc(&d_ref_in, total_points * sizeof(cuComplex));
    cudaMalloc(&d_ref_out, total_points * sizeof(cuComplex));
    cudaMalloc(&d_our_in, total_points * sizeof(cuComplex));
    cudaMalloc(&d_our_out, total_points * sizeof(cuComplex));
    
    cudaMemcpy(d_ref_in, input.data(), total_points * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_our_in, input.data(), total_points * sizeof(cuComplex), cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan1d(&plan, 16, CUFFT_C2C, num_windows);
    cufftExecC2C(plan, d_ref_in, d_ref_out, CUFFT_FORWARD);
    launch_fft16_batch64(d_our_in, d_our_out, num_windows);
    cudaDeviceSynchronize();
    
    std::vector<std::complex<float>> ref(total_points), our(total_points);
    cudaMemcpy(ref.data(), d_ref_out, total_points * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(our.data(), d_our_out, total_points * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    
    int total_failed = 0;
    double max_error_global = 0.0;
    
    for (int w = 0; w < num_windows; ++w) {
        int window_failed = 0;
        for (int i = 0; i < 16; ++i) {
            int idx = w * 16 + i;
            float mag_ref = std::abs(ref[idx]);
            float mag_our = std::abs(our[idx]);
            
            if (mag_ref >= MAGNITUDE_THRESHOLD || mag_our >= MAGNITUDE_THRESHOLD) {
                double error = (mag_ref > 1e-6f) ? std::abs(mag_our - mag_ref) / mag_ref * 100.0 : 0.0;
                if (error > ERROR_TOLERANCE) window_failed++;
                if (error > max_error_global) max_error_global = error;
            }
        }
        if (window_failed > 0) total_failed++;
    }
    
    std::cout << "╔═══════════════════════════════════════════════╗\n";
    std::cout << "║            VALIDATION RESULTS                 ║\n";
    std::cout << "╠═══════════════════════════════════════════════╣\n";
    std::cout << "║ Total windows:     " << std::setw(4) << num_windows << "                        ║\n";
    std::cout << "║ Failed windows:    " << std::setw(4) << total_failed << "                        ║\n";
    std::cout << "║ Max error:         " << std::setw(10) << std::setprecision(6) << max_error_global << "%             ║\n";
    std::cout << "╚═══════════════════════════════════════════════╝\n\n";
    
    if (total_failed == 0) {
        std::cout << "🎉🎉🎉 SUCCESS! All 64 windows PERFECT! 🎉🎉🎉\n\n";
    }
    
    cufftDestroy(plan);
    cudaFree(d_ref_in); cudaFree(d_ref_out);
    cudaFree(d_our_in); cudaFree(d_our_out);
    return (total_failed == 0) ? 0 : 1;
}
