#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

// External functions for all FFT sizes
extern "C" void launch_fft16_simple(const cuComplex*, cuComplex*, int);
extern "C" void launch_fft32_simple(const cuComplex*, cuComplex*, int);
extern "C" void launch_fft64_simple(const cuComplex*, cuComplex*, int);
extern "C" void launch_fft128_simple(const cuComplex*, cuComplex*, int);
extern "C" void launch_fft256_simple(const cuComplex*, cuComplex*, int);
extern "C" void launch_fft512_simple(const cuComplex*, cuComplex*, int);
extern "C" void launch_fft1024_simple(const cuComplex*, cuComplex*, int);

struct BenchmarkResult {
    float our_min, our_max, our_avg;
    float cufft_min, cufft_max, cufft_avg;
    float speedup;
    bool validation_passed;
};

BenchmarkResult benchmark_fft(int fft_size, int num_windows, 
                             void (*our_fft)(const cuComplex*, cuComplex*, int)) {
    const int total_points = num_windows * fft_size;
    
    // Create test signal
    std::vector<std::complex<float>> input(total_points);
    for (int w = 0; w < num_windows; ++w) {
        for (int i = 0; i < fft_size; ++i) {
            float angle = 2.0f * M_PI * i / fft_size;
            input[w * fft_size + i] = std::complex<float>(std::cos(angle), std::sin(angle));
        }
    }
    
    // Allocate GPU memory
    cuComplex *d_input, *d_our_output, *d_cufft_output;
    cudaMalloc(&d_input, total_points * sizeof(cuComplex));
    cudaMalloc(&d_our_output, total_points * sizeof(cuComplex));
    cudaMalloc(&d_cufft_output, total_points * sizeof(cuComplex));
    cudaMemcpy(d_input, input.data(), total_points * sizeof(cuComplex), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up
    our_fft(d_input, d_our_output, num_windows);
    cudaDeviceSynchronize();
    
    // Benchmark our implementation
    std::vector<float> our_times;
    for (int i = 0; i < 20; ++i) {
        cudaEventRecord(start);
        our_fft(d_input, d_our_output, num_windows);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        our_times.push_back(ms);
    }
    
    // Benchmark cuFFT
    cufftHandle plan;
    cufftPlan1d(&plan, fft_size, CUFFT_C2C, 1);
    
    std::vector<float> cufft_times;
    for (int i = 0; i < 20; ++i) {
        cudaEventRecord(start);
        for (int w = 0; w < num_windows; ++w) {
            cuComplex* in_ptr = d_input + w * fft_size;
            cuComplex* out_ptr = d_cufft_output + w * fft_size;
            cufftExecC2C(plan, in_ptr, out_ptr, CUFFT_FORWARD);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        cufft_times.push_back(ms);
    }
    
    // Calculate statistics
    std::sort(our_times.begin(), our_times.end());
    std::sort(cufft_times.begin(), cufft_times.end());
    
    BenchmarkResult result;
    result.our_min = our_times[0];
    result.our_max = our_times[our_times.size()-1];
    result.our_avg = std::accumulate(our_times.begin(), our_times.end(), 0.0f) / our_times.size();
    
    result.cufft_min = cufft_times[0];
    result.cufft_max = cufft_times[cufft_times.size()-1];
    result.cufft_avg = std::accumulate(cufft_times.begin(), cufft_times.end(), 0.0f) / cufft_times.size();
    
    result.speedup = result.cufft_avg / result.our_avg;
    
    // Quick validation
    std::vector<std::complex<float>> our_result(total_points);
    std::vector<std::complex<float>> cufft_result(total_points);
    cudaMemcpy(our_result.data(), d_our_output, total_points * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(cufft_result.data(), d_cufft_output, total_points * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    
    // Check bin 1 (should be fft_size)
    float our_bin1 = std::abs(our_result[1]);
    float cufft_bin1 = std::abs(cufft_result[1]);
    float error = std::abs(our_bin1 - cufft_bin1) / cufft_bin1 * 100.0f;
    result.validation_passed = (error < 0.1f); // 0.1% tolerance
    
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_our_output);
    cudaFree(d_cufft_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

int main() {
    std::cout << "\n=== COMPREHENSIVE FFT BENCHMARK ===\n\n";
    
    // Test configurations: {fft_size, num_windows}
    std::vector<std::pair<int, int>> configs = {
        {16, 1024},   // FFT16: 1024 windows
        {32, 512},    // FFT32: 512 windows  
        {64, 256},    // FFT64: 256 windows
        {128, 128},   // FFT128: 128 windows
        {256, 64},    // FFT256: 64 windows
        {512, 32},    // FFT512: 32 windows
        {1024, 16}    // FFT1024: 16 windows
    };
    
    std::vector<void (*)(const cuComplex*, cuComplex*, int)> functions = {
        launch_fft16_simple,
        launch_fft32_simple,
        launch_fft64_simple,
        launch_fft128_simple,
        launch_fft256_simple,
        launch_fft512_simple,
        launch_fft1024_simple
    };
    
    std::vector<BenchmarkResult> results;
    
    // Run benchmarks
    for (size_t i = 0; i < configs.size(); ++i) {
        int fft_size = configs[i].first;
        int num_windows = configs[i].second;
        
        std::cout << "Benchmarking FFT" << fft_size << " (" << num_windows << " windows)...\n";
        
        BenchmarkResult result = benchmark_fft(fft_size, num_windows, functions[i]);
        results.push_back(result);
    }
    
    // Print results table
    std::cout << "\n=== BENCHMARK RESULTS TABLE ===\n\n";
    std::cout << std::left << std::setw(8) << "FFT" 
              << std::setw(12) << "Our (ms)" 
              << std::setw(12) << "cuFFT (ms)" 
              << std::setw(10) << "Speedup" 
              << std::setw(12) << "Status" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t i = 0; i < configs.size(); ++i) {
        int fft_size = configs[i].first;
        BenchmarkResult& r = results[i];
        
        std::cout << std::left << std::setw(8) << fft_size
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.our_avg
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.cufft_avg
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.speedup
                  << std::setw(12) << (r.validation_passed ? "✅ PASS" : "❌ FAIL") << "\n";
    }
    
    // Summary
    std::cout << "\n=== SUMMARY ===\n";
    int passed = 0;
    float avg_speedup = 0.0f;
    for (const auto& r : results) {
        if (r.validation_passed) passed++;
        avg_speedup += r.speedup;
    }
    avg_speedup /= results.size();
    
    std::cout << "Validation passed: " << passed << "/" << results.size() << "\n";
    std::cout << "Average speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x\n";
    
    if (avg_speedup > 1.0f) {
        std::cout << "🎉 Our implementations are FASTER than cuFFT!\n";
    } else {
        std::cout << "📊 Our implementations are slower than cuFFT (expected for complex FFT)\n";
    }
    
    return 0;
}
