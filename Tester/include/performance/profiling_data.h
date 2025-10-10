/**
 * @file profiling_data.h
 * @brief Profiling data structures
 */

#pragma once

#include "Interface/include/signal_data.h"
#include <string>

namespace CudaCalc {

/**
 * @brief Basic profiling result (CUDA Events)
 */
struct BasicProfilingResult {
    // Timing (milliseconds)
    float upload_ms;      ///< Host → Device time
    float compute_ms;     ///< Kernel execution time
    float download_ms;    ///< Device → Host time
    float total_ms;       ///< Total time
    
    // Metadata
    std::string gpu_name;         ///< "NVIDIA GeForce RTX 3060"
    std::string cuda_version;     ///< "13.0"
    std::string driver_version;   ///< Driver version
    std::string timestamp;        ///< "2025-10-10T10:30:45"
    std::string algorithm;        ///< "FFT16_WMMA" or "FFT16_Shared2D"
    
    StrobeConfig config;          ///< Test configuration
};

} // namespace CudaCalc

