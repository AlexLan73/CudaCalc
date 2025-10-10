/**
 * @file test_speed_final.cpp
 * @brief FINAL SPEED TEST - Multiple runs for accuracy
 * 
 * Tests FFT16_WMMA_Optimized multiple times to get stable result
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft16_wmma_optimized.h"
#include "Tester/include/performance/basic_profiler.h"

using namespace CudaCalc;

int main() {
    std::cout << "=== FINAL SPEED TEST - FFT16_WMMA_Optimized ===" << std::endl;
    std::cout << "Configuration: 2D blocks [64,16] = 1024 threads, 64 FFT/block" << std::endl;
    std::cout << std::endl;
    
    try {
        // Generate signal
        SineGenerator gen(4, 1024, 8);
        auto input = gen.generate(16, false);
        
        std::cout << "Signal: " << input.signal.size() << " points, " 
                  << input.config.num_windows() << " windows" << std::endl;
        std::cout << std::endl;
        
        // Initialize
        FFT16_WMMA_Optimized fft;
        fft.initialize();
        
        // Warmup (first run always slower!)
        std::cout << "Warmup run..." << std::endl;
        auto warmup = fft.process(input);
        std::cout << "✓ Warmup complete" << std::endl;
        std::cout << std::endl;
        
        // Multiple test runs
        const int NUM_RUNS = 20;
        std::vector<float> compute_times;
        
        std::cout << "Running " << NUM_RUNS << " iterations for accuracy..." << std::endl;
        
        for (int i = 0; i < NUM_RUNS; ++i) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            // Measure ONLY compute
            cudaEventRecord(start);
            auto output = fft.process(input);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            compute_times.push_back(ms);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            if ((i+1) % 5 == 0) {
                std::cout << "  " << (i+1) << "/" << NUM_RUNS << " runs..." << std::endl;
            }
        }
        
        // Statistics
        std::sort(compute_times.begin(), compute_times.end());
        
        float min_time = compute_times.front();
        float max_time = compute_times.back();
        float median_time = compute_times[NUM_RUNS / 2];
        float avg_time = 0.0f;
        for (float t : compute_times) avg_time += t;
        avg_time /= NUM_RUNS;
        
        std::cout << std::endl;
        std::cout << "=== RESULTS (20 runs) ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Min:    " << min_time << " ms ⚡⚡⚡" << std::endl;
        std::cout << "  Median: " << median_time << " ms" << std::endl;
        std::cout << "  Mean:   " << avg_time << " ms" << std::endl;
        std::cout << "  Max:    " << max_time << " ms" << std::endl;
        std::cout << std::endl;
        
        std::cout << "=== COMPARISON WITH TARGET ===" << std::endl;
        std::cout << "  Old project: 0.007950 ms" << std::endl;
        std::cout << "  Our BEST:    " << min_time << " ms" << std::endl;
        
        if (min_time <= 0.00795f) {
            std::cout << "  Status: ✅✅✅ TARGET ACHIEVED!" << std::endl;
        } else {
            float diff_percent = ((min_time / 0.00795f) - 1.0f) * 100.0f;
            std::cout << "  Gap: +" << std::setprecision(1) << diff_percent << "%" << std::endl;
            
            if (diff_percent < 15.0f) {
                std::cout << "  Status: ✅ EXCELLENT (< 15% gap!)" << std::endl;
            } else if (diff_percent < 30.0f) {
                std::cout << "  Status: ✓ Good (< 30% gap)" << std::endl;
            } else {
                std::cout << "  Status: ⚠️  Needs more work" << std::endl;
            }
        }
        std::cout << std::endl;
        
        fft.cleanup();
        
        std::cout << "=== TEST COMPLETE ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
}

