/**
 * @file fft16_wmma.h
 * @brief FFT16 implementation using Tensor Cores (WMMA)
 * 
 * FFT for 16-point windows using Warp Matrix Multiply-Accumulate (Tensor Cores).
 * Linear unroll of 4 butterfly stages (NO for loop).
 * FP16 precision for Tensor Core operations.
 */

#pragma once

#include "Interface/include/igpu_processor.h"
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace CudaCalc {

/**
 * @brief FFT16 using Tensor Cores (WMMA, FP16)
 * 
 * Implementation details:
 * - Tensor Cores (WMMA API) for matrix operations
 * - FP16 precision (converted from FP32)
 * - Linear unroll of 4 butterfly stages
 * - FFT shift applied in kernel
 * - Optimized for Ampere architecture (sm_86, RTX 3060)
 * 
 * Performance target: < 0.05 ms compute time for 256 windows on RTX 3060
 * Expected: 5-10% faster than Shared2D due to Tensor Core acceleration
 */
class FFT16_WMMA : public IGPUProcessor {
protected:
    // Device memory
    cuComplex* d_input_ = nullptr;
    cuComplex* d_output_ = nullptr;
    
    // Configuration
    int num_windows_ = 0;
    bool initialized_ = false;
    
public:
    FFT16_WMMA();
    ~FFT16_WMMA();
    
    // IGPUProcessor interface
    bool initialize() override;
    void cleanup() override;
    OutputSpectralData process(const InputSignalData& input) override;
    
    std::string get_name() const override { return "FFT16_WMMA"; }
    std::string get_algorithm() const override { return "FFT"; }
    int get_size() const override { return 16; }
};

} // namespace CudaCalc

