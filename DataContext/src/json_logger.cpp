/**
 * @file json_logger.cpp
 * @brief JSONLogger implementation
 */

#include "DataContext/include/json_logger.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace CudaCalc {

JSONLogger::JSONLogger(const std::string& output_dir)
    : output_dir_(output_dir)
{
    // Create output directory if it doesn't exist
    if (!output_dir_.empty()) {
        fs::create_directories(output_dir_);
    }
}

bool JSONLogger::save_test_result(const TestResult& result, const std::string& filename) {
    try {
        json j;
        
        // Metadata
        j["test_name"] = result.test_name;
        j["description"] = result.description;
        j["algorithm"] = result.algorithm;
        j["timestamp"] = result.profiling.timestamp;
        j["git_commit"] = result.git_commit;
        
        // Configuration
        j["config"]["ray_count"] = result.config.ray_count;
        j["config"]["points_per_ray"] = result.config.points_per_ray;
        j["config"]["window_fft"] = result.config.window_fft;
        j["config"]["total_points"] = result.config.total_points();
        j["config"]["num_windows"] = result.config.num_windows();
        
        // Profiling
        j["profiling"]["basic"]["upload_ms"] = result.profiling.upload_ms;
        j["profiling"]["basic"]["compute_ms"] = result.profiling.compute_ms;
        j["profiling"]["basic"]["download_ms"] = result.profiling.download_ms;
        j["profiling"]["basic"]["total_ms"] = result.profiling.total_ms;
        j["profiling"]["basic"]["throughput_mpts"] = 
            result.config.total_points() / (result.profiling.total_ms * 1000.0);
        
        // GPU info
        j["profiling"]["gpu"]["name"] = result.profiling.gpu_name;
        j["profiling"]["gpu"]["cuda_version"] = result.profiling.cuda_version;
        j["profiling"]["gpu"]["driver_version"] = result.profiling.driver_version;
        
        // Validation
        j["validation"]["passed"] = result.validation.passed;
        j["validation"]["max_relative_error"] = result.validation.max_relative_error;
        j["validation"]["avg_relative_error"] = result.validation.avg_relative_error;
        j["validation"]["tolerance"] = result.validation.tolerance;
        j["validation"]["total_points"] = result.validation.total_points;
        j["validation"]["failed_points"] = result.validation.failed_points;
        j["validation"]["correct_percent"] = 
            100.0 * (result.validation.total_points - result.validation.failed_points) 
            / result.validation.total_points;
        j["validation"]["reference"] = result.validation.reference;
        
        // Save to file
        std::string full_path = output_dir_ + "/" + filename;
        std::ofstream file(full_path);
        
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << full_path << std::endl;
            return false;
        }
        
        file << std::setw(2) << j << std::endl;
        file.close();
        
        std::cout << "✓ Test result saved: " << full_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving test result: " << e.what() << std::endl;
        return false;
    }
}

bool JSONLogger::save_comparison(const std::vector<TestResult>& results, const std::string& filename) {
    try {
        json j;
        
        j["comparison_name"] = "FFT16 Performance Comparison";
        j["timestamp"] = results.empty() ? "" : results[0].profiling.timestamp;
        j["num_tests"] = results.size();
        
        // Add all test results
        json tests_array = json::array();
        
        for (const auto& result : results) {
            json test;
            test["algorithm"] = result.algorithm;
            test["compute_ms"] = result.profiling.compute_ms;
            test["total_ms"] = result.profiling.total_ms;
            test["avg_error_percent"] = result.validation.avg_relative_error * 100.0;
            test["max_error_percent"] = result.validation.max_relative_error * 100.0;
            test["validation_passed"] = result.validation.passed;
            test["correct_percent"] = 
                100.0 * (result.validation.total_points - result.validation.failed_points) 
                / result.validation.total_points;
            
            tests_array.push_back(test);
        }
        
        j["tests"] = tests_array;
        
        // Find best
        if (!results.empty()) {
            size_t fastest_idx = 0;
            float best_compute = results[0].profiling.compute_ms;
            
            for (size_t i = 1; i < results.size(); ++i) {
                if (results[i].profiling.compute_ms < best_compute) {
                    best_compute = results[i].profiling.compute_ms;
                    fastest_idx = i;
                }
            }
            
            j["best"]["fastest"] = results[fastest_idx].algorithm;
            j["best"]["compute_ms"] = results[fastest_idx].profiling.compute_ms;
            
            if (results.size() > 1) {
                float baseline = results[0].profiling.compute_ms;
                j["best"]["speedup"] = baseline / best_compute;
            }
        }
        
        // GPU info
        if (!results.empty()) {
            j["gpu"]["name"] = results[0].profiling.gpu_name;
            j["gpu"]["cuda_version"] = results[0].profiling.cuda_version;
        }
        
        // Save
        std::string full_path = output_dir_ + "/" + filename;
        std::ofstream file(full_path);
        
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << full_path << std::endl;
            return false;
        }
        
        file << std::setw(2) << j << std::endl;
        file.close();
        
        std::cout << "✓ Comparison saved: " << full_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving comparison: " << e.what() << std::endl;
        return false;
    }
}

bool JSONLogger::save_profiling(const BasicProfilingResult& profiling, const std::string& filename) {
    try {
        json j;
        
        j["algorithm"] = profiling.algorithm;
        j["timestamp"] = profiling.timestamp;
        
        j["timing"]["upload_ms"] = profiling.upload_ms;
        j["timing"]["compute_ms"] = profiling.compute_ms;
        j["timing"]["download_ms"] = profiling.download_ms;
        j["timing"]["total_ms"] = profiling.total_ms;
        
        j["gpu"]["name"] = profiling.gpu_name;
        j["gpu"]["cuda_version"] = profiling.cuda_version;
        j["gpu"]["driver_version"] = profiling.driver_version;
        
        j["config"]["ray_count"] = profiling.config.ray_count;
        j["config"]["points_per_ray"] = profiling.config.points_per_ray;
        j["config"]["window_fft"] = profiling.config.window_fft;
        
        std::string full_path = output_dir_ + "/" + filename;
        std::ofstream file(full_path);
        
        if (!file.is_open()) {
            return false;
        }
        
        file << std::setw(2) << j << std::endl;
        file.close();
        
        std::cout << "✓ Profiling saved: " << full_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

bool JSONLogger::save_validation(const ValidationResult& validation, const std::string& filename) {
    try {
        json j;
        
        j["algorithm"] = validation.algorithm;
        j["reference"] = validation.reference;
        j["passed"] = validation.passed;
        
        j["errors"]["max_relative"] = validation.max_relative_error;
        j["errors"]["max_relative_percent"] = validation.max_relative_error * 100.0;
        j["errors"]["avg_relative"] = validation.avg_relative_error;
        j["errors"]["avg_relative_percent"] = validation.avg_relative_error * 100.0;
        
        j["points"]["total"] = validation.total_points;
        j["points"]["failed"] = validation.failed_points;
        j["points"]["correct"] = validation.total_points - validation.failed_points;
        j["points"]["correct_percent"] = 
            100.0 * (validation.total_points - validation.failed_points) / validation.total_points;
        
        j["tolerance"] = validation.tolerance;
        j["tolerance_percent"] = validation.tolerance * 100.0;
        
        std::string full_path = output_dir_ + "/" + filename;
        std::ofstream file(full_path);
        
        if (!file.is_open()) {
            return false;
        }
        
        file << std::setw(2) << j << std::endl;
        file.close();
        
        std::cout << "✓ Validation saved: " << full_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace CudaCalc

