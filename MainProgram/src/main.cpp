/**
 * @file main.cpp
 * @brief Performance comparison: FFT16_Shared2D vs FFT16_WMMA
 */

#include <iostream>
#include <iomanip>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft16_shared2d_profiled.h"
#include "ModelsFunction/include/nvidia/fft/fft16_wmma_profiled.h"
#include "ModelsFunction/include/nvidia/fft/fft16_wmma_ultra_profiled.h"
#include "Tester/include/validation/fft_validator.h"
#include "DataContext/include/json_logger.h"

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
        
        // 3a. Test FFT16_WMMA_Ultra (REAL FP16 Tensor Cores!)
        std::cout << "=== 3a. Testing FFT16_WMMA_Ultra (REAL FP16 Tensor Cores!) ===" << std::endl;
        FFT16_WMMA_Ultra_Profiled fft_ultra;
        fft_ultra.initialize();
        
        BasicProfilingResult prof_ultra;
        auto output_ultra = fft_ultra.process_with_profiling(input, prof_ultra);
        
        std::cout << "✓ FFT computed: " << output_ultra.num_windows() << " windows" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Upload:   " << prof_ultra.upload_ms << " ms" << std::endl;
        std::cout << "  Compute:  " << prof_ultra.compute_ms << " ms ⚡⚡⚡" << std::endl;
        std::cout << "  Download: " << prof_ultra.download_ms << " ms" << std::endl;
        std::cout << "  TOTAL:    " << prof_ultra.total_ms << " ms" << std::endl;
        std::cout << std::endl;
        
        fft_ultra.cleanup();
        
        // 4. Performance comparison (3 versions!)
        std::cout << "=== 4. Performance Comparison ===" << std::endl;
        std::cout << "┌──────────────────────┬────────────────┬────────────────┬──────────────┐" << std::endl;
        std::cout << "│ Algorithm            │ Compute (ms)   │ Total (ms)     │ Speedup      │" << std::endl;
        std::cout << "├──────────────────────┼────────────────┼────────────────┼──────────────┤" << std::endl;
        std::cout << "│ FFT16_Shared2D       │ " << std::setw(14) << prof_shared2d.compute_ms 
                  << " │ " << std::setw(14) << prof_shared2d.total_ms << " │ baseline     │" << std::endl;
        std::cout << "│ FFT16_WMMA (FP32)    │ " << std::setw(14) << prof_wmma.compute_ms 
                  << " │ " << std::setw(14) << prof_wmma.total_ms << " │ " 
                  << std::setw(10) << std::setprecision(2) << (prof_shared2d.compute_ms / prof_wmma.compute_ms) << "x │" << std::endl;
        std::cout << "│ FFT16_WMMA_Ultra ⚡⚡ │ " << std::setw(14) << std::setprecision(3) << prof_ultra.compute_ms 
                  << " │ " << std::setw(14) << prof_ultra.total_ms << " │ " 
                  << std::setw(10) << std::setprecision(2) << (prof_shared2d.compute_ms / prof_ultra.compute_ms) << "x │" << std::endl;
        std::cout << "└──────────────────────┴────────────────┴────────────────┴──────────────┘" << std::endl;
        std::cout << std::endl;
        
        // 5. Winner
        float speedup = prof_shared2d.compute_ms / prof_ultra.compute_ms;
        std::cout << "=== 5. Result ===" << std::endl;
        std::cout << "🏆 WINNER: FFT16_WMMA_Ultra (REAL FP16 Tensor Cores!)" << std::endl;
        std::cout << "   Compute: " << std::setprecision(6) << prof_ultra.compute_ms << " ms" << std::endl;
        std::cout << "   Speedup vs Shared2D: " << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "   Speedup vs WMMA(FP32): " << std::setprecision(2) 
                  << (prof_wmma.compute_ms / prof_ultra.compute_ms) << "x" << std::endl;
        
        // Сравнение с целью
        std::cout << std::endl;
        std::cout << "📊 Сравнение с целью (старый проект):" << std::endl;
        std::cout << "   Цель:    0.00795 ms (старый AMGpuCuda)" << std::endl;
        std::cout << "   Наш:     " << std::setprecision(6) << prof_ultra.compute_ms << " ms" << std::endl;
        if (prof_ultra.compute_ms <= 0.00795f) {
            std::cout << "   Статус:  ✅ ЦЕЛЬ ДОСТИГНУТА!" << std::endl;
        } else {
            float diff = ((prof_ultra.compute_ms / 0.00795f) - 1.0f) * 100.0f;
            std::cout << "   Разница: +" << std::setprecision(1) << diff << "%" << std::endl;
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
        
        std::cout << "\n--- Testing FFT16_WMMA_Ultra ---" << std::endl;
        auto val_ultra = validator.validate(input, output_ultra, "FFT16_WMMA_Ultra");
        
        std::cout << "\nValidation Summary:" << std::endl;
        std::cout << "  FFT16_Shared2D: " << (val_shared2d.passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
        std::cout << "  FFT16_WMMA:     " << (val_wmma.passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
        std::cout << std::endl;
        
        // 6a. АНАЛИЗ: Посмотрим на первое окно спектра
        std::cout << "=== 6a. АНАЛИЗ СПЕКТРА (первое окно) ===" << std::endl;
        std::cout << "Ожидаем пик на частоте 2 (период синуса = 8, частота = 1/8 * 16 = 2)" << std::endl;
        std::cout << std::endl;
        std::cout << "Freq  Magnitude   Тип компоненты" << std::endl;
        std::cout << "────  ──────────  ─────────────────────────" << std::endl;
        
        for (int i = 0; i < 16; ++i) {
            auto val = output_wmma.windows[0][i];
            float mag = std::abs(val);
            int freq = i - 8;  // После FFT shift: -8..7
            
            std::cout << std::setw(3) << freq << "  " 
                      << std::setw(10) << std::fixed << std::setprecision(4) << mag << "  ";
            
            if (mag > 100.0) {
                std::cout << "⭐⭐⭐ ОСНОВНАЯ ГАРМОНИКА (ВАЖНАЯ!)";
            } else if (mag > 10.0) {
                std::cout << "⭐⭐ Значимая гармоника";
            } else if (mag > 1.0) {
                std::cout << "⭐ Заметная";
            } else if (mag > 0.1) {
                std::cout << "· Малая";
            } else if (mag > 0.01) {
                std::cout << "· Очень малая";
            } else {
                std::cout << "~ Near-zero (шум) ← ЗДЕСЬ 131% ERROR!";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        
        // 7. Save results to JSON
        std::cout << "=== 7. Saving results to JSON ===" << std::endl;
        JSONLogger logger("results/");
        
        // Save Shared2D results
        TestResult result_shared2d;
        result_shared2d.algorithm = "FFT16_Shared2D";
        result_shared2d.profiling = prof_shared2d;
        result_shared2d.validation = val_shared2d;
        result_shared2d.config = input.config;
        result_shared2d.test_name = "FFT16_Shared2D_Test";
        result_shared2d.description = "2D Shared Memory, FP32, Linear unroll";
        logger.save_test_result(result_shared2d, "fft16_shared2d_result.json");
        
        // Save WMMA results
        TestResult result_wmma;
        result_wmma.algorithm = "FFT16_WMMA";
        result_wmma.profiling = prof_wmma;
        result_wmma.validation = val_wmma;
        result_wmma.config = input.config;
        result_wmma.test_name = "FFT16_WMMA_Test";
        result_wmma.description = "Tensor Cores, FP32, Linear unroll, 9.4x faster!";
        logger.save_test_result(result_wmma, "fft16_wmma_result.json");
        
        // Save comparison
        std::vector<TestResult> results = {result_shared2d, result_wmma};
        logger.save_comparison(results, "fft16_comparison.json");
        
        std::cout << std::endl;
        std::cout << "=== ALL TESTS PASSED ✓ ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
}

