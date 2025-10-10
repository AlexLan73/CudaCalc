/**
 * @file fft16_wmma_ultra.h
 * @brief ULTRA-FAST FFT16 with REAL Tensor Cores (FP16)
 * 
 * Based on proven AMGpuCuda ultra_optimized_tensor_kernels
 * Target: 0.00795ms or better!
 */

#pragma once

#include "Interface/include/igpu_processor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace CudaCalc {

/**
 * @brief ULTRA-OPTIMIZED FFT16 using REAL Tensor Cores (FP16)
 * 
 * Key optimizations:
 * - FP16 precision (2x less data, REAL Tensor Cores!)
 * - 1024 threads per block (64 FFT × 16 threads)
 * - Separate real/imag arrays (SoA for coalescing)
 * - Constant memory twiddles (FP16)
 * - FP16 intrinsics (__hadd, __hsub, __hmul)
 * - 2D thread blocks [64, 16]
 * 
 * Expected: 0.005-0.008ms (40-50% faster than old WMMA!)
 */
class FFT16_WMMA_Ultra : public IGPUProcessor {
protected:
    // Device memory (FP16, separate real/imag)
    __half* d_input_real_ = nullptr;
    __half* d_input_imag_ = nullptr;
    __half* d_output_real_ = nullptr;
    __half* d_output_imag_ = nullptr;
    
    // Configuration
    int num_windows_ = 0;
    bool initialized_ = false;
    
public:
    FFT16_WMMA_Ultra();
    ~FFT16_WMMA_Ultra();
    
    // IGPUProcessor interface
    bool initialize() override;
    void cleanup() override;
    OutputSpectralData process(const InputSignalData& input) override;
    
    std::string get_name() const override { return "FFT16_WMMA_Ultra"; }
    std::string get_algorithm() const override { return "FFT"; }
    int get_size() const override { return 16; }
};

} // namespace CudaCalc

