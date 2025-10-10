/**
 * @file memory_profiler.cpp
 * @brief MemoryProfiler implementation
 */

#include "Tester/include/performance/memory_profiler.h"
#include "Interface/include/common_types.h"
#include <iostream>

namespace CudaCalc {

MemoryProfiler::MemoryProfiler(int device_id)
    : device_id_(device_id)
    , vram_free_before_(0)
    , upload_bytes_(0)
    , download_bytes_(0)
    , upload_time_ms_(0.0f)
    , download_time_ms_(0.0f)
{
}

void MemoryProfiler::start() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    
    vram_free_before_ = free;
}

void MemoryProfiler::record_upload(size_t bytes, float time_ms) {
    upload_bytes_ = bytes;
    upload_time_ms_ = time_ms;
}

void MemoryProfiler::record_download(size_t bytes, float time_ms) {
    download_bytes_ = bytes;
    download_time_ms_ = time_ms;
}

MemoryProfilingResult MemoryProfiler::get_results() {
    MemoryProfilingResult result;
    
    // Get current memory info
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    
    result.vram_total_mb = total / (1024 * 1024);
    result.vram_free_mb = free / (1024 * 1024);
    result.vram_used_mb = (vram_free_before_ - free) / (1024 * 1024);
    
    // Calculate bandwidth (GB/s)
    // Bandwidth = Bytes / Time(s) / 1e9
    if (upload_time_ms_ > 0.0f) {
        result.upload_bandwidth_gbps = (upload_bytes_ / (upload_time_ms_ / 1000.0f)) / 1e9f;
    } else {
        result.upload_bandwidth_gbps = 0.0f;
    }
    
    if (download_time_ms_ > 0.0f) {
        result.download_bandwidth_gbps = (download_bytes_ / (download_time_ms_ / 1000.0f)) / 1e9f;
    } else {
        result.download_bandwidth_gbps = 0.0f;
    }
    
    float total_bytes = upload_bytes_ + download_bytes_;
    float total_time_ms = upload_time_ms_ + download_time_ms_;
    
    if (total_time_ms > 0.0f) {
        result.total_bandwidth_gbps = (total_bytes / (total_time_ms / 1000.0f)) / 1e9f;
    } else {
        result.total_bandwidth_gbps = 0.0f;
    }
    
    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    
    result.gpu_name = std::string(prop.name);
    result.compute_capability_major = prop.major;
    result.compute_capability_minor = prop.minor;
    
    // Peak bandwidth (GB/s) - hardcoded for known GPUs
    // RTX 3060: 360 GB/s (GDDR6)
    // Can be refined with device-specific lookup table
    if (std::string(prop.name).find("3060") != std::string::npos) {
        result.peak_bandwidth_gbps = 360.0f;  // RTX 3060
    } else if (std::string(prop.name).find("3070") != std::string::npos) {
        result.peak_bandwidth_gbps = 448.0f;  // RTX 3070
    } else if (std::string(prop.name).find("3080") != std::string::npos) {
        result.peak_bandwidth_gbps = 760.0f;  // RTX 3080
    } else {
        // Estimate from memory bus width (may not be accurate)
        result.peak_bandwidth_gbps = 300.0f;  // Conservative estimate
    }
    
    // Efficiency
    if (result.peak_bandwidth_gbps > 0.0f) {
        result.bandwidth_efficiency_percent = 
            (result.total_bandwidth_gbps / result.peak_bandwidth_gbps) * 100.0f;
    } else {
        result.bandwidth_efficiency_percent = 0.0f;
    }
    
    return result;
}

} // namespace CudaCalc

