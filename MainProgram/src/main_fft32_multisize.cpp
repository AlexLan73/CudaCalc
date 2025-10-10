/**
 * @file main_fft32_multisize.cpp
 * @brief FFT32 Multi-Size Performance Test
 * 
 * Tests FFT32 V3 Final on different window counts:
 * 1, 64, 256, 1024, 4096, 16384 windows
 * 
 * Compares with AMGpuCuda targets from table:
 * | 32 | 256 | 0.010800 |
 * | 32 | 16384 | 0.048800 |
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft32_wmma_v3_final.h"
#include "Tester/include/validation/fft_validator.h"

using namespace CudaCalc;

struct TestConfig {
    int num_windows;
    double target_ms;  // From AMGpuCuda (if available)
    std::string description;
};

int main() {
    try {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════╗\n";
        std::cout << "║   FFT32 MULTI-SIZE PERFORMANCE TEST (V3)      ║\n";
        std::cout << "╚════════════════════════════════════════════════╝\n\n";

        // === TEST CONFIGURATIONS ===
        std::vector<TestConfig> configs = {
            {1,     -1.0,    "Single window"},
            {64,    -1.0,    "Small batch"},
            {256,   0.0108,  "Medium batch (from table)"},
            {1024,  -1.0,    "Large batch"},
            {4096,  -1.0,    "Very large batch"},
            {16384, 0.0488,  "Huge batch (from table)"}
        };
        
        constexpr int FFT_WINDOW = 32;
        constexpr int PERIOD = 32;  // period = wFFT (was wFFT/2)
        constexpr int NUM_RUNS = 10;  // Runs per config
        
        std::cout << "FFT window: " << FFT_WINDOW << "\n";
        std::cout << "Sine period: " << PERIOD << "\n";
        std::cout << "Runs per config: " << NUM_RUNS << "\n\n";
        
        std::cout << "┌────────┬─────────────┬────────────┬────────────┬─────────────┬──────────┐\n";
        std::cout << "│ Windows│ Total Points│ Min (ms)   │ Target (ms)│ Difference  │  Status  │\n";
        std::cout << "├────────┼─────────────┼────────────┼────────────┼─────────────┼──────────┤\n";
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // === RUN ALL CONFIGURATIONS ===
        for (const auto& config : configs) {
            const int num_windows = config.num_windows;
            const int total_points = num_windows * FFT_WINDOW;
            
            // Calculate rays and points per ray
            int ray_count = 4;
            int points_per_ray = total_points / ray_count;
            
            // Generate signal
            SineGenerator generator(ray_count, points_per_ray, PERIOD);
            auto input_data = generator.generate(FFT_WINDOW, false);
            
            // Initialize FFT
            FFT32_WMMA_V3_Final fft32;
            fft32.initialize(num_windows);
            
            // Prepare output
            std::vector<std::complex<float>> output(total_points);
            
            // Warmup
            fft32.process(input_data.signal.data(), output.data());
            
            // Multiple runs
            std::vector<float> times;
            times.reserve(NUM_RUNS);
            
            for (int run = 0; run < NUM_RUNS; ++run) {
                cudaEventRecord(start);
                fft32.process(input_data.signal.data(), output.data());
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                
                float ms = 0;
                cudaEventElapsedTime(&ms, start, stop);
                times.push_back(ms);
            }
            
            // Statistics
            std::sort(times.begin(), times.end());
            float min_time = times[0];
            
            // Output row
            std::cout << "│ " << std::setw(6) << num_windows
                      << " │ " << std::setw(11) << total_points
                      << " │ " << std::fixed << std::setprecision(5) << std::setw(10) << min_time;
            
            if (config.target_ms > 0) {
                double diff_ms = min_time - config.target_ms;
                double diff_pct = (diff_ms / config.target_ms) * 100.0;
                
                std::cout << " │ " << std::setw(10) << config.target_ms
                          << " │ " << std::setw(+7) << std::showpos << diff_pct << std::noshowpos << "%"
                          << " │ ";
                
                if (min_time <= config.target_ms) {
                    std::cout << "   ✅   ";
                } else {
                    std::cout << "   ❌   ";
                }
            } else {
                std::cout << " │     N/A    │      N/A    │   N/A    ";
            }
            
            std::cout << " │\n";
            
            fft32.cleanup();
        }
        
        std::cout << "└────────┴─────────────┴────────────┴────────────┴─────────────┴──────────┘\n\n";
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        std::cout << "═══════════════════════════════════════════════════\n";
        std::cout << "Test complete! Check results above.\n";
        std::cout << "═══════════════════════════════════════════════════\n\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

