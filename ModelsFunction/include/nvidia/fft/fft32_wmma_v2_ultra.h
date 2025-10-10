#pragma once

#include "fft32_wmma_optimized.h"

namespace CudaCalc {

class FFT32_WMMA_V2_Ultra : public FFT32_WMMA_Optimized {
public:
    FFT32_WMMA_V2_Ultra();
    virtual ~FFT32_WMMA_V2_Ultra();

    void initialize(int num_windows) override;
    void process(const void* input, void* output) override;
    void cleanup() override;

protected:
    cuComplex* d_input_;
    cuComplex* d_output_;
    int num_windows_;
    bool initialized_;
};

} // namespace CudaCalc

