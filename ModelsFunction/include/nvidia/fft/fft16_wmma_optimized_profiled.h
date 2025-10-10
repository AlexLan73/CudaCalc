/**
 * @file fft16_wmma_optimized_profiled.h
 */

#pragma once

#include "fft16_wmma_optimized.h"
#include "Tester/include/performance/basic_profiler.h"

namespace CudaCalc {

class FFT16_WMMA_Optimized_Profiled : public FFT16_WMMA_Optimized {
private:
    BasicProfiler profiler_;
    
public:
    OutputSpectralData process_with_profiling(
        const InputSignalData& input,
        BasicProfilingResult& profiling
    );
};

} // namespace CudaCalc

