/**
 * @file main_fft32.cpp
 * @brief FFT32 Performance Test (multiple runs)
 * 
 * Tests FFT32_WMMA_Optimized on 4 rays × 2048 points (256 FFT windows)
 * Target: <= 0.0108ms (from AMGpuCuda project)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft32_wmma_optimized_profiled.h"
#include "Tester/include/performance/basic_profiler.h"
#include "Tester/include/validation/fft_validator.h"
#include "DataContext/include/json_logger.h"

using namespace CudaCalc;

int main() {
    try {
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════╗\n";
        std::cout << "║   FFT32 PERFORMANCE TEST (20 RUNS)   ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n\n";

        // === CONFIGURATION ===
        constexpr int RAY_COUNT = 4;
        constexpr int POINTS_PER_RAY = 131072;  // 16384 windows × 32 / 4 rays
        constexpr int FFT_WINDOW = 32;
        constexpr int PERIOD = 32;  // period = wFFT (was wFFT/2)
        constexpr int NUM_RUNS = 20;
        constexpr double TARGET_MS = 0.0488;  // From AMGpuCuda (16384 windows)

        std::cout << "Configuration:\n";
        std::cout << "  Rays: " << RAY_COUNT << "\n";
        std::cout << "  Points per ray: " << POINTS_PER_RAY << "\n";
        std::cout << "  Total points: " << (RAY_COUNT * POINTS_PER_RAY) << "\n";
        std::cout << "  FFT window: " << FFT_WINDOW << "\n";
        std::cout << "  Sine period: " << PERIOD << "\n";
        std::cout << "  Target compute time: " << TARGET_MS << " ms\n\n";

        // === GENERATE SIGNAL ===
        std::cout << "Generating test signal...\n";
        SineGenerator generator(RAY_COUNT, POINTS_PER_RAY, PERIOD);
        auto input_data = generator.generate(FFT_WINDOW, false);
        const int num_windows = input_data.config.num_windows();
        
        std::cout << "  ✓ Generated " << num_windows << " FFT windows\n\n";

        // === INITIALIZE FFT ===
        std::cout << "Initializing FFT32_WMMA_Optimized...\n";
        FFT32_WMMA_Optimized_Profiled fft32_optimized;
        fft32_optimized.initialize(num_windows);
        std::cout << "  ✓ FFT initialized\n\n";

        // === WARMUP ===
        std::cout << "Warmup run...\n";
        std::vector<std::complex<float>> output(num_windows * FFT_WINDOW);
        BasicProfiler warmup_prof;
        fft32_optimized.process_with_profiling(
            input_data.signal.data(),
            output.data(),
            warmup_prof
        );
        auto warmup_result = warmup_prof.get_results("FFT32_WMMA_Optimized", input_data.config);
        std::cout << "  Warmup: " << std::fixed << std::setprecision(5) 
                  << warmup_result.compute_ms << " ms\n\n";

        // === MULTIPLE RUNS ===
        std::cout << "Running " << NUM_RUNS << " iterations...\n\n";
        
        std::vector<float> compute_times;
        compute_times.reserve(NUM_RUNS);

        for (int run = 0; run < NUM_RUNS; ++run) {
            BasicProfiler profiler;
            fft32_optimized.process_with_profiling(
                input_data.signal.data(),
                output.data(),
                profiler
            );

            auto result = profiler.get_results("FFT32_WMMA_Optimized", input_data.config);
            compute_times.push_back(result.compute_ms);

            if ((run + 1) % 5 == 0) {
                std::cout << "  Run " << std::setw(2) << (run + 1) << "/" << NUM_RUNS 
                          << " - Compute: " << std::fixed << std::setprecision(5) 
                          << result.compute_ms << " ms\n";
            }
        }

        // === STATISTICS ===
        std::sort(compute_times.begin(), compute_times.end());
        float min_time = compute_times[0];
        float max_time = compute_times.back();
        float median_time = compute_times[NUM_RUNS/2];
        float mean_time = std::accumulate(compute_times.begin(), compute_times.end(), 0.0f) / NUM_RUNS;

        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════╗\n";
        std::cout << "║      PERFORMANCE STATISTICS          ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n\n";
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  Min:    " << min_time << " ms\n";
        std::cout << "  Max:    " << max_time << " ms\n";
        std::cout << "  Mean:   " << mean_time << " ms\n";
        std::cout << "  Median: " << median_time << " ms\n\n";

        // === COMPARISON WITH TARGET ===
        float improvement_pct = ((TARGET_MS - min_time) / TARGET_MS) * 100.0f;
        
        std::cout << "╔═══════════════════════════════════════╗\n";
        std::cout << "║     COMPARISON WITH TARGET           ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n\n";
        std::cout << "  Target (AMGpuCuda): " << TARGET_MS << " ms\n";
        std::cout << "  Our result (MIN):   " << min_time << " ms\n";
        
        if (min_time <= TARGET_MS) {
            std::cout << "  🏆 FASTER by:       " << std::abs(improvement_pct) 
                      << "% ✅✅✅\n\n";
        } else {
            std::cout << "  ⚠️  SLOWER by:       " << std::abs(improvement_pct) 
                      << "% ❌\n\n";
        }

        // === VALIDATION ===
        std::cout << "╔═══════════════════════════════════════╗\n";
        std::cout << "║         VALIDATION CHECK             ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n\n";
        
        // Prepare OutputSpectralData
        OutputSpectralData output_data;
        output_data.windows.resize(num_windows);
        for (int w = 0; w < num_windows; ++w) {
            output_data.windows[w].resize(FFT_WINDOW);
            for (int p = 0; p < FFT_WINDOW; ++p) {
                output_data.windows[w][p] = output[w * FFT_WINDOW + p];
            }
        }
        
        FFTValidator validator(0.0001);  // 0.01% tolerance
        auto validation = validator.validate(input_data, output_data, "FFT32_WMMA_Optimized");

        std::cout << "  Average error: " << std::fixed << std::setprecision(2) 
                  << (validation.avg_relative_error * 100.0) << "%\n";
        std::cout << "  Max error:     " << (validation.max_relative_error * 100.0) << "%\n";
        std::cout << "  Failed points: " << validation.failed_points 
                  << " / " << validation.total_points 
                  << " (" << std::setprecision(1) 
                  << ((double)(validation.total_points - validation.failed_points) / validation.total_points * 100.0) 
                  << "% passed)\n";
        std::cout << "  Status: " << (validation.passed ? "✅ PASSED" : "❌ FAILED") << "\n\n";

        // === SAVE RESULTS ===
        std::cout << "Saving results to JSON...\n";
        JSONLogger logger("results");
        TestResult test_result;
        test_result.algorithm = "FFT32_WMMA_Optimized";
        test_result.profiling = warmup_result;
        test_result.validation = validation;
        test_result.config = input_data.config;
        test_result.test_name = "FFT32 Performance Test";
        logger.save_test_result(test_result, "fft32_test.json");
        std::cout << "  ✓ Results saved to results/fft32_test.json\n\n";

        // === FINAL STATUS ===
        std::cout << "╔═══════════════════════════════════════╗\n";
        std::cout << "║          FINAL STATUS                ║\n";
        std::cout << "╚═══════════════════════════════════════╝\n\n";
        
        if (validation.passed && min_time <= TARGET_MS) {
            std::cout << "  🎉 SUCCESS! FFT32 is PRODUCTION READY! 🎉\n";
            std::cout << "  ✅ Accuracy validated\n";
            std::cout << "  ✅ Performance target met\n\n";
        } else if (validation.passed) {
            std::cout << "  ⚠️  Accuracy OK, but performance needs improvement\n\n";
        } else {
            std::cout << "  ❌ Validation FAILED - accuracy needs fixing\n\n";
        }

        fft32_optimized.cleanup();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
