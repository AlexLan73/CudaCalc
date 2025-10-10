#pragma once

#include "fft32_wmma_optimized.h"

namespace CudaCalc {

/**
 * @brief FFT32 V4 CORRECT - Based on FFT16 exact butterfly pattern!
 */
class FFT32_WMMA_V4_Correct : public FFT32_WMMA_Optimized {
public:
    FFT32_WMMA_V4_Correct() = default;
    virtual ~FFT32_WMMA_V4_Correct() = default;
    
    void process(const void* input, void* output) override;
};

} // namespace CudaCalc

