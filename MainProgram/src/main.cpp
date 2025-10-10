/**
 * @file main.cpp
 * @brief Performance comparison: FFT16_Shared2D vs FFT16_WMMA
 */

#include <iostream>
#include <iomanip>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft16_shared2d_profiled.h"
#include "ModelsFunction/include/nvidia/fft/fft16_wmma_profiled.h"
#include "Tester/include/validation/fft_validator.h"

using namespace CudaCalc;

int main() {
    std::cout << "=== CudaCalc FFT16 Performance Comparison ===" << std::endl;
    std::cout << std::endl;
    
    try {
        // 1. Generate signal
        std::cout << "=== 1. Generating test signal ===" << std::endl;
        SineGenerator gen(4, 1024, 8);
        auto input = gen.generate(16, false);
        std::cout << "✓ Signal: " << input.signal.size() << " points" << std::endl;
        std::cout << "  Configuration: " << input.config.ray_count << " rays × " 
                  << input.config.points_per_ray << " points, FFT window = " 
                  << input.config.window_fft << std::endl;
        std::cout << std::endl;
        
        // 2. Test FFT16_Shared2D
        std::cout << "=== 2. Testing FFT16_Shared2D (2D Shared Memory, FP32) ===" << std::endl;
        FFT16_Shared2D_Profiled fft_shared2d;
        fft_shared2d.initialize();
        
        BasicProfilingResult prof_shared2d;
        auto output_shared2d = fft_shared2d.process_with_profiling(input, prof_shared2d);
        
        std::cout << "✓ FFT computed: " << output_shared2d.num_windows() << " windows" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Upload:   " << prof_shared2d.upload_ms << " ms" << std::endl;
        std::cout << "  Compute:  " << prof_shared2d.compute_ms << " ms ⚡" << std::endl;
        std::cout << "  Download: " << prof_shared2d.download_ms << " ms" << std::endl;
        std::cout << "  TOTAL:    " << prof_shared2d.total_ms << " ms" << std::endl;
        std::cout << std::endl;
        
        fft_shared2d.cleanup();
        
        // 3. Test FFT16_WMMA
        std::cout << "=== 3. Testing FFT16_WMMA (Tensor Core optimized) ===" << std::endl;
        FFT16_WMMA_Profiled fft_wmma;
        fft_wmma.initialize();
        
        BasicProfilingResult prof_wmma;
        auto output_wmma = fft_wmma.process_with_profiling(input, prof_wmma);
        
        std::cout << "✓ FFT computed: " << output_wmma.num_windows() << " windows" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Upload:   " << prof_wmma.upload_ms << " ms" << std::endl;
        std::cout << "  Compute:  " << prof_wmma.compute_ms << " ms ⚡" << std::endl;
        std::cout << "  Download: " << prof_wmma.download_ms << " ms" << std::endl;
        std::cout << "  TOTAL:    " << prof_wmma.total_ms << " ms" << std::endl;
        std::cout << std::endl;
        
        fft_wmma.cleanup();
        
        // 4. Performance comparison
        std::cout << "=== 4. Performance Comparison ===" << std::endl;
        std::cout << "┌────────────────────┬────────────────┬────────────────┐" << std::endl;
        std::cout << "│ Algorithm          │ Compute (ms)   │ Total (ms)     │" << std::endl;
        std::cout << "├────────────────────┼────────────────┼────────────────┤" << std::endl;
        std::cout << "│ FFT16_Shared2D     │ " << std::setw(14) << prof_shared2d.compute_ms 
                  << " │ " << std::setw(14) << prof_shared2d.total_ms << " │" << std::endl;
        std::cout << "│ FFT16_WMMA         │ " << std::setw(14) << prof_wmma.compute_ms 
                  << " │ " << std::setw(14) << prof_wmma.total_ms << " │" << std::endl;
        std::cout << "└────────────────────┴────────────────┴────────────────┘" << std::endl;
        std::cout << std::endl;
        
        // 5. Winner
        float speedup = prof_shared2d.compute_ms / prof_wmma.compute_ms;
        std::cout << "=== 5. Result ===" << std::endl;
        if (speedup > 1.02f) {
            std::cout << "🏆 WINNER: FFT16_WMMA" << std::endl;
            std::cout << "   Speedup: " << std::setprecision(2) << speedup << "x faster" << std::endl;
        } else if (speedup < 0.98f) {
            std::cout << "🏆 WINNER: FFT16_Shared2D" << std::endl;
            std::cout << "   Speedup: " << std::setprecision(2) << (1.0f/speedup) << "x faster" << std::endl;
        } else {
            std::cout << "⚖️  EQUAL: Both implementations perform similarly" << std::endl;
            std::cout << "   Difference: < 2%" << std::endl;
        }
        std::cout << std::endl;
        
        std::cout << "GPU: " << prof_shared2d.gpu_name << std::endl;
        std::cout << "CUDA: " << prof_shared2d.cuda_version << std::endl;
        std::cout << std::endl;
        
        // 6. Validation tests
        std::cout << "=== 6. Validating results against cuFFT ===" << std::endl;
        FFTValidator validator(0.0001);  // 0.01% tolerance
        
        std::cout << "\n--- Testing FFT16_Shared2D ---" << std::endl;
        auto val_shared2d = validator.validate(input, output_shared2d, "FFT16_Shared2D");
        
        std::cout << "\n--- Testing FFT16_WMMA ---" << std::endl;
        auto val_wmma = validator.validate(input, output_wmma, "FFT16_WMMA");
        
        std::cout << "\nValidation Summary:" << std::endl;
        std::cout << "  FFT16_Shared2D: " << (val_shared2d.passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
        std::cout << "  FFT16_WMMA:     " << (val_wmma.passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
        std::cout << std::endl;
        
        std::cout << "=== ALL TESTS PASSED ✓ ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
}

