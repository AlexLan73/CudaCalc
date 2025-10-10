/**
 * @file memory_profiler.h
 * @brief Memory profiler for GPU metrics
 * 
 * Collects VRAM usage, bandwidth, and GPU utilization.
 */

#pragma once

#include <string>
#include <cuda_runtime.h>

namespace CudaCalc {

/**
 * @brief Memory profiling result
 */
struct MemoryProfilingResult {
    // VRAM
    size_t vram_total_mb;      ///< Total GPU memory (MB)
    size_t vram_free_mb;       ///< Free memory before (MB)
    size_t vram_used_mb;       ///< Used by this operation (MB)
    
    // Bandwidth (estimated)
    float upload_bandwidth_gbps;    ///< Upload bandwidth (GB/s)
    float download_bandwidth_gbps;  ///< Download bandwidth (GB/s)
    float total_bandwidth_gbps;     ///< Total bandwidth (GB/s)
    
    // GPU info
    std::string gpu_name;
    int compute_capability_major;
    int compute_capability_minor;
    
    // Peak theoretical bandwidth
    float peak_bandwidth_gbps;
    float bandwidth_efficiency_percent;  ///< Actual / Peak * 100
};

/**
 * @brief Memory profiler
 * 
 * Measures VRAM and bandwidth usage.
 * 
 * Usage:
 * @code
 * MemoryProfiler prof;
 * prof.start();
 * 
 * // ... upload data ...
 * prof.record_upload(bytes_uploaded, upload_time_ms);
 * 
 * // ... download data ...
 * prof.record_download(bytes_downloaded, download_time_ms);
 * 
 * auto result = prof.get_results();
 * @endcode
 */
class MemoryProfiler {
private:
    size_t vram_free_before_;
    size_t upload_bytes_;
    size_t download_bytes_;
    float upload_time_ms_;
    float download_time_ms_;
    
    int device_id_;
    
public:
    MemoryProfiler(int device_id = 0);
    
    /**
     * @brief Start profiling (record initial VRAM)
     */
    void start();
    
    /**
     * @brief Record upload operation
     * @param bytes Bytes uploaded
     * @param time_ms Time in milliseconds
     */
    void record_upload(size_t bytes, float time_ms);
    
    /**
     * @brief Record download operation
     * @param bytes Bytes downloaded
     * @param time_ms Time in milliseconds
     */
    void record_download(size_t bytes, float time_ms);
    
    /**
     * @brief Get profiling results
     */
    MemoryProfilingResult get_results();
};

} // namespace CudaCalc

