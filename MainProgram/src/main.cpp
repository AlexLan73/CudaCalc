/**
 * @file main.cpp
 * @brief Test FFT16_Shared2D with profiling
 */

#include <iostream>
#include <iomanip>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft16_shared2d_profiled.h"

using namespace CudaCalc;

int main() {
    std::cout << "=== CudaCalc FFT16 Profiling Test ===" << std::endl;
    std::cout << std::endl;
    
    try {
        // 1. Generate signal
        std::cout << "=== 1. Generating signal ===" << std::endl;
        SineGenerator gen(4, 1024, 8);
        auto input = gen.generate(16, false);
        std::cout << "✓ Signal: " << input.signal.size() << " points" << std::endl;
        std::cout << std::endl;
        
        // 2. Run with profiling
        std::cout << "=== 2. Running FFT16_Shared2D with profiling ===" << std::endl;
        FFT16_Shared2D_Profiled fft;
        fft.initialize();
        
        BasicProfilingResult profiling;
        auto output = fft.process_with_profiling(input, profiling);
        
        std::cout << "✓ FFT computed: " << output.num_windows() << " windows" << std::endl;
        std::cout << std::endl;
        
        // 3. Show profiling results
        std::cout << "=== 3. Profiling Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Upload:   " << profiling.upload_ms << " ms" << std::endl;
        std::cout << "  Compute:  " << profiling.compute_ms << " ms" << std::endl;
        std::cout << "  Download: " << profiling.download_ms << " ms" << std::endl;
        std::cout << "  ─────────────────────" << std::endl;
        std::cout << "  TOTAL:    " << profiling.total_ms << " ms" << std::endl;
        std::cout << std::endl;
        
        std::cout << "  GPU:       " << profiling.gpu_name << std::endl;
        std::cout << "  CUDA:      " << profiling.cuda_version << std::endl;
        std::cout << "  Algorithm: " << profiling.algorithm << std::endl;
        std::cout << "  Timestamp: " << profiling.timestamp << std::endl;
        std::cout << std::endl;
        
        // 4. Performance metrics
        std::cout << "=== 4. Performance Metrics ===" << std::endl;
        int total_points = input.signal.size();
        float compute_gflops = (5.0f * total_points * std::log2(16)) / (profiling.compute_ms * 1e6);
        std::cout << "  Throughput: " << (total_points / (profiling.total_ms * 1000.0)) << " Mpts/s" << std::endl;
        std::cout << "  Compute:    ~" << compute_gflops << " GFLOPS (approx)" << std::endl;
        std::cout << std::endl;
        
        fft.cleanup();
        
        std::cout << "=== TEST PASSED ✓ ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
}

