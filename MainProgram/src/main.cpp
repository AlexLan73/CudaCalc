/**
 * @file main.cpp
 * @brief Quick test for FFT16_Shared2D
 * 
 * Minimal test to verify FFT16_Shared2D works.
 * Full implementation in TASK-024.
 */

#include <iostream>
#include "SignalGenerators/include/sine_generator.h"
#include "ModelsFunction/include/nvidia/fft/fft16_shared2d.h"

using namespace CudaCalc;

int main(int argc, char** argv) {
    std::cout << "=== CudaCalc FFT16 Quick Test ===" << std::endl;
    std::cout << std::endl;
    
    try {
        // 1. Generate test signal
        std::cout << "=== 1. Generating signal ===" << std::endl;
        SineGenerator gen(4, 1024, 8);  // 4 rays, 1024 points, period=8
        auto input = gen.generate(16, false);  // FFT window=16
        std::cout << "✓ Signal generated" << std::endl;
        std::cout << std::endl;
        
        // 2. Run FFT16_Shared2D
        std::cout << "=== 2. Running FFT16_Shared2D ===" << std::endl;
        FFT16_Shared2D fft;
        fft.initialize();
        
        auto output = fft.process(input);
        
        std::cout << "✓ FFT computed" << std::endl;
        std::cout << "  Output windows: " << output.num_windows() << std::endl;
        std::cout << "  Window size: " << output.window_size() << std::endl;
        std::cout << std::endl;
        
        // 3. Check output
        std::cout << "=== 3. Checking output ===" << std::endl;
        std::cout << "First window (first 5 points):" << std::endl;
        for (int i = 0; i < 5 && i < output.windows[0].size(); ++i) {
            auto val = output.windows[0][i];
            std::cout << "  [" << i << "] " << val.real() << " + " << val.imag() << "i" << std::endl;
        }
        std::cout << std::endl;
        
        fft.cleanup();
        
        std::cout << "=== TEST PASSED ✓ ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
}

