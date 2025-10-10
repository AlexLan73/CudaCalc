#pragma once

#include "fft32_wmma_optimized.h"

namespace CudaCalc {

/**
 * @brief FFT32 V3 FINAL - Based on FFT16 Optimized success!
 */
class FFT32_WMMA_V3_Final : public FFT32_WMMA_Optimized {
public:
    FFT32_WMMA_V3_Final() = default;
    virtual ~FFT32_WMMA_V3_Final() = default;
    
    // Override process to use V3 kernel
    void process(const void* input, void* output) override;
};

} // namespace CudaCalc

