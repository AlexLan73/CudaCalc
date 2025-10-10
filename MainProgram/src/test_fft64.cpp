/**
 * @file test_fft64.cpp
 * @brief Quick test FFT64 (based on FFT16 winner pattern!)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft64_wmma_optimized.h"
#include "Tester/include/validation/fft_validator.h"

using namespace CudaCalc;

int main() {
    try {
        std::cout << "\n╔═══════════════════════════════════════╗\n";
        std::cout << "║    FFT64 QUICK TEST (256 windows)    ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n\n";

        // Generate signal: 256 windows × 64 points = 16384 points
        // 4 rays × 4096 points per ray = 16384 total
        SineGenerator gen(4, 4096, 64);  // period = wFFT
        auto input = gen.generate(64, false);
        
        FFT64_WMMA_Optimized fft64;
        fft64.initialize(256);
        
        std::vector<std::complex<float>> output(256 * 64);
        
        // Warmup
        fft64.process(input.signal.data(), output.data());
        
        // Performance test (10 runs)
        std::vector<float> times;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        for (int i = 0; i < 10; ++i) {
            cudaEventRecord(start);
            fft64.process(input.signal.data(), output.data());
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times.push_back(ms);
        }
        
        std::sort(times.begin(), times.end());
        float min_time = times[0];
        float median_time = times[5];
        float mean_time = 0.0f;
        for (float t : times) mean_time += t;
        mean_time /= times.size();
        
        std::cout << "Performance (10 runs):\n";
        std::cout << "  Min:    " << std::fixed << std::setprecision(5) << min_time << " ms\n";
        std::cout << "  Median: " << median_time << " ms\n";
        std::cout << "  Mean:   " << mean_time << " ms\n";
        std::cout << "  Target: 0.010 ms (from AMGpuCuda FFT64×256)\n";
        
        float diff = ((min_time / 0.010f) - 1.0f) * 100.0f;
        if (min_time <= 0.010f) {
            std::cout << "  Status: ✅ FASTER by " << std::abs(diff) << "%!\n\n";
        } else {
            std::cout << "  Status: ⚠️ Slower by " << diff << "%\n\n";
        }
        
        // Validation
        OutputSpectralData output_data;
        output_data.windows.resize(256);
        for (int w = 0; w < 256; ++w) {
            output_data.windows[w].resize(64);
            for (int p = 0; p < 64; ++p) {
                output_data.windows[w][p] = output[w * 64 + p];
            }
        }
        
        FFTValidator validator(0.0001);
        auto val = validator.validate(input, output_data, "FFT64");
        
        std::cout << "Validation:\n";
        std::cout << "  Avg error: " << std::setprecision(2) << (val.avg_relative_error * 100.0) << "%\n";
        std::cout << "  Max error: " << (val.max_relative_error * 100.0) << "%\n";
        std::cout << "  Failed: " << val.failed_points << " / " << val.total_points << "\n";
        std::cout << "  Status: " << (val.passed ? "✅ PASSED!" : "❌ FAILED") << "\n\n";
        
        if (val.passed && min_time <= 0.010f) {
            std::cout << "🎉🎉🎉 FFT64 is EXCELLENT! Faster AND accurate! 🎉🎉🎉\n\n";
        } else if (val.passed) {
            std::cout << "✅ Accuracy OK! Speed: " << (min_time <= 0.010f ? "FAST!" : "needs tuning") << "\n\n";
        } else if (min_time <= 0.010f) {
            std::cout << "⚡ Speed OK! Accuracy needs debugging.\n\n";
        } else {
            std::cout << "⚠️ Both speed and accuracy need work.\n\n";
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        fft64.cleanup();
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

