/**
 * @file signal_types.h
 * @brief Signal type enumeration
 * 
 * Defines types of signals that can be generated for testing.
 */

#pragma once

namespace CudaCalc {

/**
 * @brief Types of test signals
 * 
 * Enum for different signal generators.
 * Expandable for future signal types.
 */
enum class SignalType {
    SINE,           ///< Sinusoidal signal (current implementation)
    QUADRATURE,     ///< Quadrature signal (future)
    MODULATED,      ///< Modulated signal (future)
    PULSE_MOD,      ///< Pulse-modulated signal (future)
    GAUSSIAN_NOISE, ///< Gaussian noise (future)
    REFLECTED,      ///< Reflected signal like interference (future)
    CUSTOM          ///< Custom user-defined signal
};

/**
 * @brief Convert SignalType to string
 */
inline const char* signal_type_to_string(SignalType type) {
    switch (type) {
        case SignalType::SINE: return "SINE";
        case SignalType::QUADRATURE: return "QUADRATURE";
        case SignalType::MODULATED: return "MODULATED";
        case SignalType::PULSE_MOD: return "PULSE_MOD";
        case SignalType::GAUSSIAN_NOISE: return "GAUSSIAN_NOISE";
        case SignalType::REFLECTED: return "REFLECTED";
        case SignalType::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

} // namespace CudaCalc

