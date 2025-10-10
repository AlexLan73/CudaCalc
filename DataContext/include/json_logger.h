/**
 * @file json_logger.h
 * @brief JSON logger for test results
 * 
 * Saves profiling and validation results to JSON files.
 */

#pragma once

#include "Interface/include/signal_data.h"
#include "Tester/include/performance/profiling_data.h"
#include "Tester/include/validation/validation_result.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace CudaCalc {

/**
 * @brief Complete test result
 */
struct TestResult {
    std::string algorithm;
    BasicProfilingResult profiling;
    ValidationResult validation;
    StrobeConfig config;
    
    // Optional metadata
    std::string test_name;
    std::string description;
    std::string git_commit;
};

/**
 * @brief JSON logger for test results
 * 
 * Usage:
 * @code
 * JSONLogger logger("results/");
 * 
 * TestResult result;
 * result.algorithm = "FFT16_WMMA";
 * result.profiling = profiling_data;
 * result.validation = validation_data;
 * 
 * logger.save_test_result(result, "fft16_wmma_test.json");
 * @endcode
 */
class JSONLogger {
private:
    std::string output_dir_;
    
public:
    /**
     * @brief Constructor
     * @param output_dir Output directory for JSON files
     */
    explicit JSONLogger(const std::string& output_dir);
    
    /**
     * @brief Save single test result
     * @param result Test result to save
     * @param filename Output filename
     * @return true if saved successfully
     */
    bool save_test_result(const TestResult& result, const std::string& filename);
    
    /**
     * @brief Save comparison of multiple tests
     * @param results Vector of test results
     * @param filename Output filename
     * @return true if saved successfully
     */
    bool save_comparison(const std::vector<TestResult>& results, const std::string& filename);
    
    /**
     * @brief Save profiling result only
     */
    bool save_profiling(const BasicProfilingResult& profiling, const std::string& filename);
    
    /**
     * @brief Save validation result only
     */
    bool save_validation(const ValidationResult& validation, const std::string& filename);
    
    /**
     * @brief Set output directory
     */
    void set_output_dir(const std::string& dir) { output_dir_ = dir; }
    
    /**
     * @brief Get output directory
     */
    std::string get_output_dir() const { return output_dir_; }
};

} // namespace CudaCalc

