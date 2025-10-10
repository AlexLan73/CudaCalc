/**
 * @file main_fft32_comparison_table.cpp
 * @brief FFT32 Full Comparison Table (All Sizes)
 * 
 * Tests FFT32 on different window counts to compare with AMGpuCuda table
 * Table row: | 32 | 256 | 0.010800 | 0.964850 | 89.34 | ULTRA |
 * Table row: | 32 | 16384 | 0.048800 | 52.652700 | 1079.15 | ULTRA |
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft32_wmma_optimized_profiled.h"
#include "Tester/include/performance/basic_profiler.h"

using namespace CudaCalc;

int main() {
    try {
        std::cout << "\n";
        std::cout << "╔═════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║            FFT32 COMPLETE COMPARISON TABLE                             ║\n";
        std::cout << "╚═════════════════════════════════════════════════════════════════════════╝\n\n";

        // === TEST SIZES (matching old project table) ===
        std::vector<int> window_counts = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
        
        // Targets from old project table
        std::vector<double> targets = {
            0.008050, 0.006150, 0.006450, 0.006450, 0.006250, 0.006600,
            0.083950, 0.007700, 0.010800, 0.014300, 0.044300, 0.022600,
            0.030100, 0.046400, 0.048800
        };
        
        constexpr int FFT_WINDOW = 32;
        constexpr int PERIOD = 16;
        constexpr int NUM_RUNS = 10;
        
        std::cout << "Configuration:\n";
        std::cout << "  FFT window: " << FFT_WINDOW << "\n";
        std::cout << "  Sine period: " << PERIOD << "\n";
        std::cout << "  Runs per size: " << NUM_RUNS << "\n\n";
        
        std::cout << "┌─────────┬──────────────┬─────────────┬──────────────┬──────────────┬──────────┐\n";
        std::cout << "│ Windows │ Total Points │ Our MIN (ms)│ Target (ms)  │ Diff (ms)    │ Diff (%) │\n";
        std::cout << "├─────────┼──────────────┼─────────────┼──────────────┼──────────────┼──────────┤\n";
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // === RUN ALL SIZES ===
        for (size_t i = 0; i < window_counts.size(); ++i) {
            const int num_windows = window_counts[i];
            const double target = targets[i];
            const int total_points = num_windows * FFT_WINDOW;
            
            // Calculate rays and points per ray
            int ray_count = (num_windows <= 4) ? 1 : 4;
            int points_per_ray = total_points / ray_count;
            
            // Generate signal
            SineGenerator generator(ray_count, points_per_ray, PERIOD);
            auto input_data = generator.generate(FFT_WINDOW, false);
            
            // Initialize FFT
            FFT32_WMMA_Optimized_Profiled fft32;
            fft32.initialize(num_windows);
            
            // Prepare output
            std::vector<std::complex<float>> output(total_points);
            
            // Warmup
            BasicProfiler warmup_prof;
            fft32.process_with_profiling(input_data.signal.data(), output.data(), warmup_prof);
            
            // Multiple runs
            std::vector<float> times;
            times.reserve(NUM_RUNS);
            
            for (int run = 0; run < NUM_RUNS; ++run) {
                BasicProfiler profiler;
                fft32.process_with_profiling(input_data.signal.data(), output.data(), profiler);
                auto result = profiler.get_results("FFT32", input_data.config);
                times.push_back(result.compute_ms);
            }
            
            // Statistics
            std::sort(times.begin(), times.end());
            float min_time = times[0];
            
            // Calculate difference
            double diff_ms = min_time - target;
            double diff_pct = (diff_ms / target) * 100.0;
            
            // Output row
            std::cout << "│ " << std::setw(7) << num_windows
                      << " │ " << std::setw(12) << total_points
                      << " │ " << std::fixed << std::setprecision(6) << std::setw(11) << min_time
                      << " │ " << std::setw(12) << target
                      << " │ " << std::showpos << std::setw(12) << diff_ms << std::noshowpos
                      << " │ " << std::showpos << std::setw(7) << std::setprecision(1) << diff_pct << "%" << std::noshowpos
                      << " │\n";
            
            fft32.cleanup();
        }
        
        std::cout << "└─────────┴──────────────┴─────────────┴──────────────┴──────────────┴──────────┘\n\n";
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        std::cout << "═══════════════════════════════════════════════════════════════════════════\n";
        std::cout << "Legend:\n";
        std::cout << "  Target: AMGpuCuda REAL FFT results (from table)\n";
        std::cout << "  Our:    CudaCalc COMPLEX FFT results\n";
        std::cout << "  Note:   COMPLEX FFT is computationally heavier than REAL FFT!\n";
        std::cout << "═══════════════════════════════════════════════════════════════════════════\n\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

