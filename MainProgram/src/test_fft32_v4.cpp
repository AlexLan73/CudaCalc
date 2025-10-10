/**
 * @file test_fft32_v4.cpp
 * @brief Quick test for FFT32 V4 (correct butterfly)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft32_wmma_v4_correct.h"
#include "Tester/include/validation/fft_validator.h"
#include "Tester/include/performance/basic_profiler.h"

using namespace CudaCalc;

int main() {
    try {
        std::cout << "\n=== FFT32 V4 CORRECT TEST (256 windows) ===\n\n";

        // Generate signal: 256 windows
        SineGenerator gen(4, 2048, 16);
        auto input = gen.generate(32, false);
        const int num_windows = 256;
        
        // Initialize FFT
        FFT32_WMMA_V4_Correct fft32;
        fft32.initialize(num_windows);
        
        // Test
        std::vector<std::complex<float>> output(num_windows * 32);
        
        // Warmup
        fft32.process(input.signal.data(), output.data());
        
        // Performance test (10 runs)
        std::vector<float> times;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        for (int i = 0; i < 10; ++i) {
            cudaEventRecord(start);
            fft32.process(input.signal.data(), output.data());
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times.push_back(ms);
        }
        
        std::sort(times.begin(), times.end());
        float min_time = times[0];
        
        std::cout << "Performance:\n";
        std::cout << "  Min: " << std::fixed << std::setprecision(5) << min_time << " ms\n";
        std::cout << "  Target: 0.0108 ms\n";
        
        if (min_time <= 0.0108f) {
            std::cout << "  Status: ✅ FASTER!\n\n";
        } else {
            float diff = ((min_time / 0.0108f) - 1.0f) * 100.0f;
            std::cout << "  Status: ⚠️ +" << std::setprecision(1) << diff << "% slower\n\n";
        }
        
        // Validation
        std::cout << "Validation:\n";
        OutputSpectralData output_data;
        output_data.windows.resize(num_windows);
        for (int w = 0; w < num_windows; ++w) {
            output_data.windows[w].resize(32);
            for (int p = 0; p < 32; ++p) {
                output_data.windows[w][p] = output[w * 32 + p];
            }
        }
        
        FFTValidator validator(0.0001);
        auto validation = validator.validate(input, output_data, "FFT32_V4");
        
        std::cout << "  Avg error: " << std::setprecision(2) 
                  << (validation.avg_relative_error * 100.0) << "%\n";
        std::cout << "  Max error: " << (validation.max_relative_error * 100.0) << "%\n";
        std::cout << "  Status: " << (validation.passed ? "✅ PASSED" : "❌ FAILED") << "\n\n";
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        fft32.cleanup();
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

