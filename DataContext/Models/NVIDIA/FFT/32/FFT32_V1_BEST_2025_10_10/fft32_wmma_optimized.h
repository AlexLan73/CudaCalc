#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace CudaCalc {

class IGPUProcessor {
public:
    virtual ~IGPUProcessor() = default;
    virtual void initialize(int num_windows) = 0;
    virtual void process(const void* input, void* output) = 0;
    virtual void cleanup() = 0;
};

class FFT32_WMMA_Optimized : public IGPUProcessor {
public:
    FFT32_WMMA_Optimized();
    virtual ~FFT32_WMMA_Optimized();

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

