/**
 * @file fft16_shared2d_profiled.h
 * @brief FFT16_Shared2D with integrated profiling
 * 
 * Extension of FFT16_Shared2D with BasicProfiler integration.
 */

#pragma once

#include "fft16_shared2d.h"
#include "Tester/include/performance/basic_profiler.h"

namespace CudaCalc {

/**
 * @brief FFT16_Shared2D with profiling support
 * 
 * Same as FFT16_Shared2D but returns profiling results.
 */
class FFT16_Shared2D_Profiled : public FFT16_Shared2D {
private:
    BasicProfiler profiler_;
    
public:
    /**
     * @brief Process with profiling
     * @param input Input signal
     * @param[out] profiling Profiling results
     * @return Output spectral data
     */
    OutputSpectralData process_with_profiling(
        const InputSignalData& input,
        BasicProfilingResult& profiling
    );
};

} // namespace CudaCalc

