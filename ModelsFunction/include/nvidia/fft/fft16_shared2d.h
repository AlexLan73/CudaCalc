/**
 * @file fft16_shared2d.h
 * @brief FFT16 implementation using 2D Shared Memory
 * 
 * FFT for 16-point windows using 2D shared memory organization.
 * Linear unroll of 4 butterfly stages (NO for loop) for maximum speed.
 * FP32 precision.
 */

#pragma once

#include "Interface/include/igpu_processor.h"
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace CudaCalc {

/**
 * @brief FFT16 using 2D Shared Memory (FP32)
 * 
 * Implementation details:
 * - 64 FFT in one block (1024 threads = 64 × 16)
 * - Shared memory: [64 FFTs][16 points] 2D array
 * - Linear unroll of 4 butterfly stages (NO for loop!)
 * - FFT shift applied in kernel
 * - FP32 precision
 * 
 * Performance target: < 1.0 ms compute time for 256 windows on RTX 3060
 */
class FFT16_Shared2D : public IGPUProcessor {
protected:
    // Device memory
    cuComplex* d_input_ = nullptr;
    cuComplex* d_output_ = nullptr;
    
    // Configuration
    int num_windows_ = 0;
    bool initialized_ = false;
    
public:
    FFT16_Shared2D();
    ~FFT16_Shared2D();
    
    // IGPUProcessor interface
    bool initialize() override;
    void cleanup() override;
    OutputSpectralData process(const InputSignalData& input) override;
    
    std::string get_name() const override { return "FFT16_Shared2D"; }
    std::string get_algorithm() const override { return "FFT"; }
    int get_size() const override { return 16; }
};

} // namespace CudaCalc

