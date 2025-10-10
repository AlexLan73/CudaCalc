#pragma once

#include "fft32_wmma_optimized.h"

namespace CudaCalc {

class BasicProfiler;  // Forward declaration

class FFT32_WMMA_Optimized_Profiled : public FFT32_WMMA_Optimized {
public:
    void process_with_profiling(const void* input, void* output, BasicProfiler& profiler);
};

} // namespace CudaCalc

