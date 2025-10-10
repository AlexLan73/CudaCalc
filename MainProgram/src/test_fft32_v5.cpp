/**
 * @file test_fft32_v5.cpp
 * @brief Quick test FFT32 V5 (fixed twiddles)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft32_wmma_v5_fixed.h"
#include "Tester/include/validation/fft_validator.h"

using namespace CudaCalc;

int main() {
    try {
        std::cout << "\n╔═══════════════════════════════════════╗\n";
        std::cout << "║  FFT32 V5 FIXED TEST (256 windows)   ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n\n";

        // 256 windows test
        SineGenerator gen(4, 2048, 32);  // period = wFFT (was wFFT/2)
        auto input = gen.generate(32, false);
        
        FFT32_WMMA_V5_Fixed fft32;
        fft32.initialize(256);
        
        std::vector<std::complex<float>> output(256 * 32);
        
        // Warmup
        fft32.process(input.signal.data(), output.data());
        
        // Performance (10 runs)
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
        
        std::cout << "Performance:\n";
        std::cout << "  Min: " << std::fixed << std::setprecision(5) << times[0] << " ms\n";
        std::cout << "  Target: 0.0108 ms\n";
        float diff = ((times[0] / 0.0108f) - 1.0f) * 100.0f;
        if (times[0] <= 0.0108f) {
            std::cout << "  Status: ✅ FASTER by " << std::abs(diff) << "%!\n\n";
        } else {
            std::cout << "  Status: ⚠️ Slower by " << diff << "%\n\n";
        }
        
        // Validation
        OutputSpectralData output_data;
        output_data.windows.resize(256);
        for (int w = 0; w < 256; ++w) {
            output_data.windows[w].resize(32);
            for (int p = 0; p < 32; ++p) {
                output_data.windows[w][p] = output[w * 32 + p];
            }
        }
        
        FFTValidator validator(0.0001);
        auto val = validator.validate(input, output_data, "FFT32_V5");
        
        std::cout << "Validation:\n";
        std::cout << "  Avg error: " << std::setprecision(2) << (val.avg_relative_error * 100.0) << "%\n";
        std::cout << "  Max error: " << (val.max_relative_error * 100.0) << "%\n";
        std::cout << "  Failed: " << val.failed_points << " / " << val.total_points << "\n";
        std::cout << "  Status: " << (val.passed ? "✅ PASSED!" : "❌ FAILED") << "\n\n";
        
        if (val.passed && times[0] <= 0.0108f) {
            std::cout << "🎉🎉🎉 SUCCESS! FFT32 V5 is PERFECT! 🎉🎉🎉\n\n";
        } else if (val.passed) {
            std::cout << "✅ Accuracy OK! Speed needs minor optimization.\n\n";
        } else {
            std::cout << "❌ Still has accuracy issues. Need more fixes.\n\n";
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        fft32.cleanup();
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

