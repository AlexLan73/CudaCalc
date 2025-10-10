/**
 * @file common_types.h
 * @brief Common types and constants for CudaCalc
 * 
 * Shared type definitions, constants, and utility macros.
 * Part of Interface module (header-only).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace CudaCalc {

// ===================================================================
// CUDA ERROR CHECKING
// ===================================================================

/**
 * @brief Check CUDA error and throw exception if failed
 * 
 * Usage: CUDA_CHECK(cudaMemcpy(...));
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA Error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err) \
            ); \
        } \
    } while(0)

/**
 * @brief Check cuFFT error and throw exception if failed
 */
#define CUFFT_CHECK(call) \
    do { \
        cufftResult_t err = call; \
        if (err != CUFFT_SUCCESS) { \
            throw std::runtime_error( \
                std::string("cuFFT Error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - Error code: " + std::to_string(err) \
            ); \
        } \
    } while(0)

// ===================================================================
// CONSTANTS
// ===================================================================

/// Mathematical constant PI
constexpr float kPI = 3.14159265358979323846f;

/// Mathematical constant 2*PI
constexpr float k2PI = 6.28318530717958647692f;

/// Default tolerance for floating point comparisons
constexpr float kDefaultTolerance = 1e-6f;

/// Default validation tolerance (0.01%)
constexpr double kDefaultValidationTolerance = 0.0001;

// ===================================================================
// GPU INFORMATION
// ===================================================================

/**
 * @brief Get GPU device name
 * @param device_id Device ID (default: 0)
 * @return GPU name string
 */
inline std::string get_gpu_name(int device_id = 0) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    
    if (err != cudaSuccess) {
        return "Unknown GPU";
    }
    
    return std::string(prop.name);
}

/**
 * @brief Get CUDA driver version
 * @return Driver version string
 */
inline std::string get_cuda_driver_version() {
    int driver_version = 0;
    cudaError_t err = cudaDriverGetVersion(&driver_version);
    
    if (err != cudaSuccess) {
        return "Unknown";
    }
    
    int major = driver_version / 1000;
    int minor = (driver_version % 1000) / 10;
    
    return std::to_string(major) + "." + std::to_string(minor);
}

/**
 * @brief Get CUDA runtime version
 */
inline std::string get_cuda_runtime_version() {
    int runtime_version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
    
    if (err != cudaSuccess) {
        return "Unknown";
    }
    
    int major = runtime_version / 1000;
    int minor = (runtime_version % 1000) / 10;
    
    return std::to_string(major) + "." + std::to_string(minor);
}

/**
 * @brief Get compute capability
 */
inline std::string get_compute_capability(int device_id = 0) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    
    if (err != cudaSuccess) {
        return "Unknown";
    }
    
    return std::to_string(prop.major) + "." + std::to_string(prop.minor);
}

} // namespace CudaCalc

