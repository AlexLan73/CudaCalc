/**
 * @file sine_generator.h
 * @brief Sinusoidal signal generator
 * 
 * Generates complex sinusoidal signals for FFT testing.
 */

#pragma once

#include "base_generator.h"
#include <cmath>

namespace CudaCalc {

/**
 * @brief Generator for sinusoidal signals
 * 
 * Generates complex sine wave: signal[n] = amplitude * exp(i * 2π * n / period)
 * 
 * Example usage:
 * @code
 * SineGenerator gen(4, 1024, 8);  // 4 rays, 1024 points/ray, period=8
 * auto input = gen.generate(16, true);  // FFT window=16, save for validation
 * @endcode
 */
class SineGenerator : public BaseGenerator {
private:
    int period_;        ///< Period in points (e.g., 8 for FFT16 test)
    float amplitude_;   ///< Signal amplitude (default: 1.0)
    float phase_;       ///< Initial phase in radians (default: 0.0)
    
public:
    /**
     * @brief Construct sine generator
     * @param ray_count Number of rays (e.g., 4)
     * @param points_per_ray Points per ray (e.g., 1024)
     * @param period Period in points (e.g., 8 = half of FFT16 window)
     * @param amplitude Amplitude (default: 1.0)
     * @param phase Initial phase in radians (default: 0.0)
     */
    SineGenerator(int ray_count, int points_per_ray, int period,
                  float amplitude = 1.0f, float phase = 0.0f);
    
    /**
     * @brief Generate sinusoidal signal
     * @param window_fft FFT window size (e.g., 16)
     * @param return_for_validation Save for Python validation?
     * @return InputSignalData with generated sine signal
     * 
     * Generates: signal[n] = amplitude * (cos(angle) + i*sin(angle))
     * where angle = 2π * n / period + phase
     */
    InputSignalData generate(int window_fft, bool return_for_validation = false) override;
    
    /**
     * @brief Get signal type
     */
    SignalType get_type() const override { return SignalType::SINE; }
    
    // Getters
    int get_period() const { return period_; }
    float get_amplitude() const { return amplitude_; }
    float get_phase() const { return phase_; }
    
    // Setters (for advanced usage)
    void set_period(int period) { period_ = period; }
    void set_amplitude(float amplitude) { amplitude_ = amplitude; }
    void set_phase(float phase) { phase_ = phase; }
};

} // namespace CudaCalc

