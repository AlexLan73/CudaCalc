#pragma once

#include "fft32_wmma_optimized.h"

namespace CudaCalc {

class FFT32_WMMA_V5_Fixed : public FFT32_WMMA_Optimized {
public:
    void process(const void* input, void* output) override;
};

} // namespace CudaCalc

