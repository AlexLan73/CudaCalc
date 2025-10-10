/**
 * @file basic_profiler.cpp
 * @brief BasicProfiler implementation
 */

#include "Tester/include/performance/basic_profiler.h"
#include "Interface/include/common_types.h"
#include <chrono>
#include <iomanip>
#include <sstream>

namespace CudaCalc {

BasicProfiler::BasicProfiler() {
    CUDA_CHECK(cudaEventCreate(&start_upload_));
    CUDA_CHECK(cudaEventCreate(&end_upload_));
    CUDA_CHECK(cudaEventCreate(&start_compute_));
    CUDA_CHECK(cudaEventCreate(&end_compute_));
    CUDA_CHECK(cudaEventCreate(&start_download_));
    CUDA_CHECK(cudaEventCreate(&end_download_));
}

BasicProfiler::~BasicProfiler() {
    cudaEventDestroy(start_upload_);
    cudaEventDestroy(end_upload_);
    cudaEventDestroy(start_compute_);
    cudaEventDestroy(end_compute_);
    cudaEventDestroy(start_download_);
    cudaEventDestroy(end_download_);
}

void BasicProfiler::start_upload() {
    CUDA_CHECK(cudaEventRecord(start_upload_));
}

void BasicProfiler::end_upload() {
    CUDA_CHECK(cudaEventRecord(end_upload_));
    upload_recorded_ = true;
}

void BasicProfiler::start_compute() {
    CUDA_CHECK(cudaEventRecord(start_compute_));
}

void BasicProfiler::end_compute() {
    CUDA_CHECK(cudaEventRecord(end_compute_));
    compute_recorded_ = true;
}

void BasicProfiler::start_download() {
    CUDA_CHECK(cudaEventRecord(start_download_));
}

void BasicProfiler::end_download() {
    CUDA_CHECK(cudaEventRecord(end_download_));
    download_recorded_ = true;
}

BasicProfilingResult BasicProfiler::get_results(const std::string& algorithm, const StrobeConfig& config) {
    // Synchronize to ensure all events are recorded
    CUDA_CHECK(cudaEventSynchronize(end_download_));
    
    BasicProfilingResult result;
    
    // Calculate elapsed times
    if (upload_recorded_) {
        CUDA_CHECK(cudaEventElapsedTime(&result.upload_ms, start_upload_, end_upload_));
    }
    
    if (compute_recorded_) {
        CUDA_CHECK(cudaEventElapsedTime(&result.compute_ms, start_compute_, end_compute_));
    }
    
    if (download_recorded_) {
        CUDA_CHECK(cudaEventElapsedTime(&result.download_ms, start_download_, end_download_));
    }
    
    result.total_ms = result.upload_ms + result.compute_ms + result.download_ms;
    
    // Metadata
    result.algorithm = algorithm;
    result.gpu_name = get_gpu_name();
    result.cuda_version = get_cuda_runtime_version();
    result.driver_version = get_cuda_driver_version();
    result.config = config;
    
    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now = *std::localtime(&time_t_now);
    
    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%S");
    result.timestamp = oss.str();
    
    return result;
}

} // namespace CudaCalc

