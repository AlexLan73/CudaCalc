/**
 * @file fft16_wmma_profiled.h
 * @brief FFT16_WMMA with integrated profiling
 */

#pragma once

#include "fft16_wmma.h"
#include "Tester/include/performance/basic_profiler.h"

namespace CudaCalc {

/**
 * @brief FFT16_WMMA with profiling support
 */
class FFT16_WMMA_Profiled : public FFT16_WMMA {
private:
    BasicProfiler profiler_;
    
public:
    /**
     * @brief Process with profiling
     */
    OutputSpectralData process_with_profiling(
        const InputSignalData& input,
        BasicProfilingResult& profiling
    );
};

} // namespace CudaCalc

