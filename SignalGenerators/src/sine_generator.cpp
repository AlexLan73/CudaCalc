/**
 * @file sine_generator.cpp
 * @brief Implementation of sinusoidal signal generator
 */

#include "SignalGenerators/include/sine_generator.h"
#include "Interface/include/common_types.h"
#include <cmath>
#include <iostream>

namespace CudaCalc {

SineGenerator::SineGenerator(int ray_count, int points_per_ray, int period,
                             float amplitude, float phase)
    : BaseGenerator(ray_count, points_per_ray)
    , period_(period)
    , amplitude_(amplitude)
    , phase_(phase)
{
    if (period <= 0) {
        throw std::invalid_argument("Period must be positive");
    }
    if (amplitude <= 0.0f) {
        throw std::invalid_argument("Amplitude must be positive");
    }
}

InputSignalData SineGenerator::generate(int window_fft, bool return_for_validation) {
    InputSignalData data;
    
    // Configuration
    data.config.ray_count = ray_count_;
    data.config.points_per_ray = points_per_ray_;
    data.config.window_fft = window_fft;
    data.return_for_validation = return_for_validation;
    
    // Calculate total points
    int total = data.config.total_points();
    data.signal.resize(total);
    
    // Generate sine wave for entire strobe
    // Formula: signal[n] = amplitude * exp(i * 2π * n / period + i * phase)
    //                    = amplitude * (cos(angle) + i * sin(angle))
    for (int n = 0; n < total; ++n) {
        float angle = k2PI * static_cast<float>(n) / static_cast<float>(period_) + phase_;
        
        data.signal[n] = std::complex<float>(
            amplitude_ * std::cos(angle),  // Real part
            amplitude_ * std::sin(angle)   // Imaginary part
        );
    }
    
    // Debug output
    std::cout << "SineGenerator: Generated " << total << " points" << std::endl;
    std::cout << "  Ray count: " << ray_count_ << std::endl;
    std::cout << "  Points per ray: " << points_per_ray_ << std::endl;
    std::cout << "  Period: " << period_ << " points" << std::endl;
    std::cout << "  Amplitude: " << amplitude_ << std::endl;
    std::cout << "  Phase: " << phase_ << " rad" << std::endl;
    std::cout << "  FFT window: " << window_fft << std::endl;
    std::cout << "  Number of windows: " << data.config.num_windows() << std::endl;
    std::cout << "  Return for validation: " << (return_for_validation ? "YES" : "NO") << std::endl;
    
    return data;
}

} // namespace CudaCalc

