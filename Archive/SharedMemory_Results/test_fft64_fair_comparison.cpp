#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

extern "C" void launch_fft64_shared_memory_v2(const cuComplex*, cuComplex*, int);
extern "C" void launch_fft64_simple_single(const cuComplex*, cuComplex*, int);

int main() {
    std::cout << "\n=== FFT64 FAIR COMPARISON: Shared Memory vs Simple Single ===\n\n";
    
    const int num_windows = 1024;
    const int points_per_window = 64;
    const int total_points = num_windows * points_per_window;
    
    // Generate test signal: exp(2πi*n/64) for each window
    std::vector<cuComplex> input(total_points);
    for (int w = 0; w < num_windows; ++w) {
        for (int n = 0; n < points_per_window; ++n) {
            float angle = 2.0f * M_PI * n / points_per_window;
            input[w * points_per_window + n] = make_cuComplex(cosf(angle), sinf(angle));
        }
    }
    
    // Allocate GPU memory
    cuComplex *d_input, *d_output_shared, *d_output_simple, *d_cufft_output;
    cudaMalloc(&d_input, total_points * sizeof(cuComplex));
    cudaMalloc(&d_output_shared, total_points * sizeof(cuComplex));
    cudaMalloc(&d_output_simple, total_points * sizeof(cuComplex));
    cudaMalloc(&d_cufft_output, total_points * sizeof(cuComplex));
    
    cudaMemcpy(d_input, input.data(), total_points * sizeof(cuComplex), cudaMemcpyHostToDevice);
    
    // Test shared memory version
    std::cout << "Testing FFT64 Shared Memory v2...\n";
    launch_fft64_shared_memory_v2(d_input, d_output_shared, num_windows);
    cudaDeviceSynchronize();
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        std::cout << "FFT64 Shared Memory kernel failed: " << cudaGetErrorString(cuda_err) << "\n";
        return 1;
    }
    
    // Test simple single version
    std::cout << "Testing FFT64 Simple Single...\n";
    launch_fft64_simple_single(d_input, d_output_simple, num_windows);
    cudaDeviceSynchronize();
    
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        std::cout << "FFT64 Simple Single kernel failed: " << cudaGetErrorString(cuda_err) << "\n";
        return 1;
    }
    
    // Test cuFFT for reference
    std::cout << "Testing cuFFT...\n";
    cufftHandle plan;
    cufftPlan1d(&plan, points_per_window, CUFFT_C2C, num_windows);
    cufftExecC2C(plan, (cufftComplex*)d_input, (cufftComplex*)d_cufft_output, CUFFT_FORWARD);
    cufftDestroy(plan);
    cudaDeviceSynchronize();
    
    // Copy results back
    std::vector<cuComplex> output_shared(total_points);
    std::vector<cuComplex> output_simple(total_points);
    std::vector<cuComplex> output_cufft(total_points);
    
    cudaMemcpy(output_shared.data(), d_output_shared, total_points * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_simple.data(), d_output_simple, total_points * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_cufft.data(), d_cufft_output, total_points * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    
    // Validation
    const double MAGNITUDE_THRESHOLD = 0.01;
    const double ERROR_TOLERANCE = 0.01; // 0.01%
    
    std::cout << "\n=== SHARED MEMORY VALIDATION ===\n";
    int shared_errors = 0;
    double max_shared_error = 0.0;
    
    for (int i = 0; i < total_points; ++i) {
        double mag_shared = sqrt(output_shared[i].x * output_shared[i].x + output_shared[i].y * output_shared[i].y);
        double mag_cufft = sqrt(output_cufft[i].x * output_cufft[i].x + output_cufft[i].y * output_cufft[i].y);
        
        if (mag_cufft >= MAGNITUDE_THRESHOLD || mag_shared >= MAGNITUDE_THRESHOLD) {
            double error = std::abs(mag_shared - mag_cufft) / mag_cufft * 100.0;
            if (error > ERROR_TOLERANCE) {
                shared_errors++;
                max_shared_error = std::max(max_shared_error, error);
            }
        }
    }
    
    std::cout << "Total bins checked: " << total_points << "\n";
    std::cout << "Bins with errors: " << shared_errors << "\n";
    std::cout << "Max error: " << max_shared_error << "%\n";
    if (shared_errors == 0) {
        std::cout << "✅ SHARED MEMORY VALIDATION PASSED!\n";
    } else {
        std::cout << "❌ SHARED MEMORY VALIDATION FAILED!\n";
    }
    
    std::cout << "\n=== SIMPLE SINGLE VALIDATION ===\n";
    int simple_errors = 0;
    double max_simple_error = 0.0;
    
    for (int i = 0; i < total_points; ++i) {
        double mag_simple = sqrt(output_simple[i].x * output_simple[i].x + output_simple[i].y * output_simple[i].y);
        double mag_cufft = sqrt(output_cufft[i].x * output_cufft[i].x + output_cufft[i].y * output_cufft[i].y);
        
        if (mag_cufft >= MAGNITUDE_THRESHOLD || mag_simple >= MAGNITUDE_THRESHOLD) {
            double error = std::abs(mag_simple - mag_cufft) / mag_cufft * 100.0;
            if (error > ERROR_TOLERANCE) {
                simple_errors++;
                max_simple_error = std::max(max_simple_error, error);
            }
        }
    }
    
    std::cout << "Total bins checked: " << total_points << "\n";
    std::cout << "Bins with errors: " << simple_errors << "\n";
    std::cout << "Max error: " << max_simple_error << "%\n";
    if (simple_errors == 0) {
        std::cout << "✅ SIMPLE SINGLE VALIDATION PASSED!\n";
    } else {
        std::cout << "❌ SIMPLE SINGLE VALIDATION FAILED!\n";
    }
    
    // Show first window results
    std::cout << "\n=== FIRST WINDOW RESULTS COMPARISON ===\n";
    std::cout << "Bin\tShared\t\tSimple\t\tcuFFT\n";
    for (int i = 0; i < 8; ++i) {
        printf("%d\t%.6f\t%.6f\t%.6f\n", i, 
               sqrt(output_shared[i].x * output_shared[i].x + output_shared[i].y * output_shared[i].y),
               sqrt(output_simple[i].x * output_simple[i].x + output_simple[i].y * output_simple[i].y),
               sqrt(output_cufft[i].x * output_cufft[i].x + output_cufft[i].y * output_cufft[i].y));
    }
    
    // Performance comparison
    std::cout << "\n=== PERFORMANCE COMPARISON ===\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up
    launch_fft64_shared_memory_v2(d_input, d_output_shared, num_windows);
    launch_fft64_simple_single(d_input, d_output_simple, num_windows);
    cudaDeviceSynchronize();
    
    // Time shared memory version
    float best_shared = 1e9f;
    for (int i = 0; i < 20; ++i) {
        cudaEventRecord(start);
        launch_fft64_shared_memory_v2(d_input, d_output_shared, num_windows);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_shared) best_shared = ms;
    }
    
    // Time simple single version
    float best_simple = 1e9f;
    for (int i = 0; i < 20; ++i) {
        cudaEventRecord(start);
        launch_fft64_simple_single(d_input, d_output_simple, num_windows);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_simple) best_simple = ms;
    }
    
    // Time cuFFT
    cufftHandle plan2;
    cufftPlan1d(&plan2, points_per_window, CUFFT_C2C, num_windows);
    float best_cufft = 1e9f;
    for (int i = 0; i < 20; ++i) {
        cudaEventRecord(start);
        cufftExecC2C(plan2, (cufftComplex*)d_input, (cufftComplex*)d_cufft_output, CUFFT_FORWARD);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_cufft) best_cufft = ms;
    }
    cufftDestroy(plan2);
    
    std::cout << "Shared Memory: " << best_shared << " ms\n";
    std::cout << "Simple Single: " << best_simple << " ms\n";
    std::cout << "cuFFT:         " << best_cufft << " ms\n";
    std::cout << "Speedup vs Simple: " << (best_simple / best_shared) << "x\n";
    std::cout << "Speedup vs cuFFT:  " << (best_cufft / best_shared) << "x\n";
    
    // Summary
    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Shared Memory: " << (shared_errors == 0 ? "✅ PASSED" : "❌ FAILED") << "\n";
    std::cout << "Simple Single: " << (simple_errors == 0 ? "✅ PASSED" : "❌ FAILED") << "\n";
    if (shared_errors == 0 && best_shared < best_simple) {
        std::cout << "🎉 Shared Memory optimization successful!\n";
    } else if (shared_errors == 0 && best_shared == best_simple) {
        std::cout << "✅ Shared Memory works but no speedup\n";
    } else if (shared_errors == 0) {
        std::cout << "⚠️  Shared Memory slower than Simple\n";
    } else {
        std::cout << "❌ Shared Memory optimization needs fixes\n";
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output_shared);
    cudaFree(d_output_simple);
    cudaFree(d_cufft_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
