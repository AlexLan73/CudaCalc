/**
 * @file base_generator.h
 * @brief Base class for signal generators
 * 
 * Abstract base for all signal generator implementations.
 */

#pragma once

#include "Interface/include/signal_data.h"
#include "signal_types.h"

namespace CudaCalc {

/**
 * @brief Abstract base class for signal generators
 * 
 * All signal generators inherit from this class.
 * Provides common interface for signal generation.
 */
class BaseGenerator {
protected:
    int ray_count_;
    int points_per_ray_;
    
public:
    BaseGenerator(int ray_count, int points_per_ray)
        : ray_count_(ray_count), points_per_ray_(points_per_ray) {}
    
    virtual ~BaseGenerator() = default;
    
    /**
     * @brief Generate signal
     * @param window_fft FFT window size
     * @param return_for_validation Flag to save for Python validation
     * @return InputSignalData with generated signal
     */
    virtual InputSignalData generate(int window_fft, bool return_for_validation = false) = 0;
    
    /**
     * @brief Get signal type
     */
    virtual SignalType get_type() const = 0;
    
    /**
     * @brief Get ray count
     */
    int get_ray_count() const { return ray_count_; }
    
    /**
     * @brief Get points per ray
     */
    int get_points_per_ray() const { return points_per_ray_; }
    
    /**
     * @brief Get total points
     */
    int get_total_points() const { return ray_count_ * points_per_ray_; }
};

} // namespace CudaCalc

