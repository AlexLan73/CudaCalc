/**
 * @file fft_validator.h
 * @brief FFT result validator using cuFFT reference
 */

#pragma once

#include "validation_result.h"
#include "Interface/include/signal_data.h"
#include <cufft.h>

namespace CudaCalc {

/**
 * @brief FFT validator using cuFFT as reference
 * 
 * Compares FFT results against cuFFT implementation.
 * Uses configurable relative error tolerance.
 * 
 * Usage:
 * @code
 * FFTValidator validator(0.0001);  // 0.01% tolerance
 * auto result = validator.validate(input, output, "FFT16_WMMA");
 * if (result.passed) {
 *     std::cout << "✓ Validation passed" << std::endl;
 * }
 * @endcode
 */
class FFTValidator {
private:
    double tolerance_;               ///< Relative error tolerance
    cufftHandle cufft_plan_;         ///< cuFFT plan
    bool plan_initialized_ = false;
    
public:
    /**
     * @brief Constructor
     * @param tolerance Relative error tolerance (default: 0.0001 = 0.01%)
     */
    explicit FFTValidator(double tolerance = 0.0001);
    
    ~FFTValidator();
    
    /**
     * @brief Validate FFT results against cuFFT
     * @param input Input signal data
     * @param output Output spectral data to validate
     * @param algorithm Algorithm name (for logging)
     * @return Validation result
     */
    ValidationResult validate(
        const InputSignalData& input,
        const OutputSpectralData& output,
        const std::string& algorithm
    );
    
    /**
     * @brief Set tolerance
     */
    void set_tolerance(double tolerance) { tolerance_ = tolerance; }
    
    /**
     * @brief Get tolerance
     */
    double get_tolerance() const { return tolerance_; }
};

} // namespace CudaCalc

