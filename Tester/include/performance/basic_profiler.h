/**
 * @file basic_profiler.h
 * @brief Basic GPU profiler using CUDA Events
 * 
 * Measures upload, compute, and download times separately.
 */

#pragma once

#include "profiling_data.h"
#include <cuda_runtime.h>
#include <functional>

namespace CudaCalc {

/**
 * @brief Basic profiler using CUDA Events
 * 
 * Measures three phases:
 * 1. Upload: Host → Device (cudaMemcpy)
 * 2. Compute: Kernel execution
 * 3. Download: Device → Host (cudaMemcpy)
 * 
 * Usage:
 * @code
 * BasicProfiler profiler;
 * profiler.start_upload();
 * // ... cudaMemcpy H→D ...
 * profiler.end_upload();
 * 
 * profiler.start_compute();
 * // ... kernel launch ...
 * profiler.end_compute();
 * 
 * profiler.start_download();
 * // ... cudaMemcpy D→H ...
 * profiler.end_download();
 * 
 * auto results = profiler.get_results("FFT16_Shared2D", config);
 * @endcode
 */
class BasicProfiler {
private:
    cudaEvent_t start_upload_, end_upload_;
    cudaEvent_t start_compute_, end_compute_;
    cudaEvent_t start_download_, end_download_;
    
    bool upload_recorded_ = false;
    bool compute_recorded_ = false;
    bool download_recorded_ = false;
    
public:
    BasicProfiler();
    ~BasicProfiler();
    
    // Upload timing
    void start_upload();
    void end_upload();
    
    // Compute timing
    void start_compute();
    void end_compute();
    
    // Download timing
    void start_download();
    void end_download();
    
    /**
     * @brief Get profiling results
     * @param algorithm Algorithm name
     * @param config Strobe configuration
     * @return Profiling result with all timings
     */
    BasicProfilingResult get_results(const std::string& algorithm, const StrobeConfig& config);
};

} // namespace CudaCalc

