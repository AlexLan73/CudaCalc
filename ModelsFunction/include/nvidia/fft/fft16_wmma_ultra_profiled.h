/**
 * @file fft16_wmma_ultra_profiled.h
 * @brief FFT16_WMMA_Ultra with profiling
 */

#pragma once

#include "fft16_wmma_ultra.h"
#include "Tester/include/performance/basic_profiler.h"

namespace CudaCalc {

class FFT16_WMMA_Ultra_Profiled : public FFT16_WMMA_Ultra {
private:
    BasicProfiler profiler_;
    
public:
    OutputSpectralData process_with_profiling(
        const InputSignalData& input,
        BasicProfilingResult& profiling
    );
};

} // namespace CudaCalc

