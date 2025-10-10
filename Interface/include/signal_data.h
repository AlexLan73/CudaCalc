/**
 * @file signal_data.h
 * @brief Signal data structures for CudaCalc
 * 
 * Defines core data structures for signal input and configuration.
 * Part of Interface module (header-only).
 */

#pragma once

#include <vector>
#include <complex>
#include <cstddef>

namespace CudaCalc {

/**
 * @brief Configuration of signal strobe
 * 
 * Strobe = basic data unit consisting of k rays with n points each.
 */
struct StrobeConfig {
    int ray_count;          ///< Number of rays (e.g., 4)
    int points_per_ray;     ///< Points per ray (e.g., 1024)
    int window_fft;         ///< FFT window size (e.g., 16)
    
    /**
     * @brief Calculate total number of points in strobe
     * @return ray_count * points_per_ray
     */
    int total_points() const {
        return ray_count * points_per_ray;
    }
    
    /**
     * @brief Calculate number of FFT windows
     * @return total_points / window_fft
     */
    int num_windows() const {
        return total_points() / window_fft;
    }
};

/**
 * @brief Input signal data (HOST memory, CPU)
 * 
 * Contains signal data to be processed on GPU.
 * Device memory is managed internally by GPU processor implementations.
 */
struct InputSignalData {
    /// Signal data: complex points (e.g., 4096 for 4×1024 strobe)
    std::vector<std::complex<float>> signal;
    
    /// Configuration
    StrobeConfig config;
    
    /// Flag: save data for Python validation?
    bool return_for_validation = false;
    
    /**
     * @brief Check if signal size matches configuration
     */
    bool is_valid() const {
        return signal.size() == static_cast<size_t>(config.total_points());
    }
    
    /**
     * @brief Get signal size
     */
    size_t size() const {
        return signal.size();
    }
};

/**
 * @brief Output spectral data (HOST memory, CPU)
 * 
 * Contains FFT results organized by windows.
 * Clean interface without device pointers.
 */
struct OutputSpectralData {
    /**
     * @brief Spectral windows
     * 
     * Format: windows[window_index][spectrum_index]
     * Example: For FFT16 with 256 windows: windows[0..255][0..15]
     */
    std::vector<std::vector<std::complex<float>>> windows;
    
    /**
     * @brief Get number of windows
     */
    size_t num_windows() const {
        return windows.size();
    }
    
    /**
     * @brief Get window size (number of spectrum points)
     */
    size_t window_size() const {
        return windows.empty() ? 0 : windows[0].size();
    }
    
    /**
     * @brief Check if data is valid (all windows same size)
     */
    bool is_valid() const {
        if (windows.empty()) return false;
        
        size_t expected_size = windows[0].size();
        for (const auto& window : windows) {
            if (window.size() != expected_size) return false;
        }
        return true;
    }
};

} // namespace CudaCalc

