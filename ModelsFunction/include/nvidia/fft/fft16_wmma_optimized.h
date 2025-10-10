/**
 * @file fft16_wmma_optimized.h
 * @brief MICRO-OPTIMIZED FFT16 WMMA (FP32)
 * 
 * Target: 0.00795ms or better!
 */

#pragma once

#include "fft16_wmma.h"

namespace CudaCalc {

class FFT16_WMMA_Optimized : public FFT16_WMMA {
public:
    FFT16_WMMA_Optimized();
    ~FFT16_WMMA_Optimized() = default;
    
    OutputSpectralData process(const InputSignalData& input) override;
    
    std::string get_name() const override { return "FFT16_WMMA_Optimized"; }
};

} // namespace CudaCalc

