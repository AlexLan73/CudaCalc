/**
 * @file validation_result.h
 * @brief Validation result structures
 */

#pragma once

#include <string>

namespace CudaCalc {

/**
 * @brief Validation result
 */
struct ValidationResult {
    bool passed;                 ///< Did validation pass?
    double max_relative_error;   ///< Maximum relative error
    double avg_relative_error;   ///< Average relative error
    double tolerance;            ///< Tolerance used
    int total_points;            ///< Total points validated
    int failed_points;           ///< Number of points that failed
    
    std::string algorithm;       ///< Algorithm name
    std::string reference;       ///< Reference implementation (e.g., "cuFFT")
    
    /**
     * @brief Check if validation passed
     */
    bool is_valid() const { return passed; }
};

} // namespace CudaCalc

