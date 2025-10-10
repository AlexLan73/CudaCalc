#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace CudaCalc {

class FFT64_WMMA_Optimized {
public:
    FFT64_WMMA_Optimized() = default;
    ~FFT64_WMMA_Optimized() = default;

    void initialize(int num_windows);
    void process(const void* input, void* output);
    void cleanup();

protected:
    cuComplex* d_input_ = nullptr;
    cuComplex* d_output_ = nullptr;
    int num_windows_ = 0;
    bool initialized_ = false;
};

} // namespace CudaCalc

