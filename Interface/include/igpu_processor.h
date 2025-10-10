/**
 * @file igpu_processor.h
 * @brief Base interface for GPU signal processors
 * 
 * Defines abstract interface that all GPU implementations must follow.
 * Part of Interface module (header-only).
 */

#pragma once

#include "signal_data.h"
#include <string>

namespace CudaCalc {

/**
 * @brief Abstract base class for GPU signal processors
 * 
 * All GPU implementations (FFT, IFFT, Correlation, etc.) inherit from this.
 * Provides unified interface for initialization, processing, and cleanup.
 */
class IGPUProcessor {
public:
    virtual ~IGPUProcessor() = default;
    
    /**
     * @brief Initialize GPU resources
     * @return true if successful, false otherwise
     * 
     * Should allocate device memory, create CUDA handles, etc.
     * Must be called before process().
     */
    virtual bool initialize() = 0;
    
    /**
     * @brief Clean up GPU resources
     * 
     * Should free device memory, destroy CUDA handles, etc.
     * Called automatically in destructor (RAII).
     */
    virtual void cleanup() = 0;
    
    /**
     * @brief Process input signal and produce output
     * @param input Input signal data
     * @return Output spectral data
     * @throws std::runtime_error if processing fails
     * @throws std::invalid_argument if input invalid
     * 
     * Main processing function. Must be called after initialize().
     */
    virtual OutputSpectralData process(const InputSignalData& input) = 0;
    
    /**
     * @brief Get processor name/identifier
     * @return Name string (e.g., "FFT16_WMMA", "FFT16_Shared2D")
     */
    virtual std::string get_name() const = 0;
    
    /**
     * @brief Get algorithm type
     * @return Algorithm string (e.g., "FFT", "IFFT", "Correlation")
     */
    virtual std::string get_algorithm() const = 0;
    
    /**
     * @brief Get processing size (window size for FFT)
     * @return Size (e.g., 16 for FFT16)
     */
    virtual int get_size() const = 0;
};

} // namespace CudaCalc

