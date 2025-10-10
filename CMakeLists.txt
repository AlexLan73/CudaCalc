cmake_minimum_required(VERSION 3.20)

# ===================================================================
# CUDA COMPILER (MUST BE BEFORE project())
# ===================================================================
# Use CUDA 13.0 from /usr/local/cuda-13.0
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(EXISTS "/usr/local/cuda-13.0/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc")
    endif()
endif()

# ===================================================================
# CUDA ARCHITECTURE (MUST BE BEFORE project())
# ===================================================================
# Target GPU: RTX 3060 (Ampere, Compute Capability 8.6)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

# ===================================================================
# PROJECT
# ===================================================================
project(CudaCalc VERSION 0.1.0 LANGUAGES CXX CUDA)

# ===================================================================
# PROJECT DESCRIPTION
# ===================================================================
# CudaCalc - Production-ready GPU primitives library for signal processing
# Focus: Maximum performance FFT, IFFT, Correlation, Convolution
# Target: NVIDIA RTX 3060 (Ampere, sm_86), CUDA 13.x
# Platforms: Ubuntu Linux, Windows

# ===================================================================
# C++ STANDARD
# ===================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ===================================================================
# CUDA SETUP
# ===================================================================
find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architecture: sm_${CMAKE_CUDA_ARCHITECTURES}")

# ===================================================================
# DEPENDENCIES
# ===================================================================

# JSON library (for logging and configuration)
find_package(nlohmann_json 3.11.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(json)
endif()

# Google Test (optional, for unit tests)
find_package(GTest QUIET)
if(GTest_FOUND)
    enable_testing()
    message(STATUS "Google Test found - unit tests enabled")
else()
    message(STATUS "Google Test not found - unit tests disabled")
endif()

# ===================================================================
# COMPILER FLAGS
# ===================================================================

# === C++ FLAGS ===
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Release optimization
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

# Debug information
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")

# === CUDA FLAGS ===
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Release optimization for CUDA
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")

# Debug for CUDA (full debug symbols)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0 -DDEBUG")

# Separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ===================================================================
# BUILD TYPE
# ===================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "Build type not specified, defaulting to Release")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ===================================================================
# INCLUDE DIRECTORIES
# ===================================================================
include_directories(${CMAKE_SOURCE_DIR})

# ===================================================================
# OUTPUT DIRECTORIES
# ===================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ===================================================================
# SUBDIRECTORIES (MODULES)
# ===================================================================
# Order matters! Dependencies must be built first.

message(STATUS "=== Building CudaCalc Modules ===")

# 1. Interface (no dependencies, header-only)
add_subdirectory(Interface)
message(STATUS "  [1/6] Interface")

# 2. SignalGenerators (depends: Interface)
add_subdirectory(SignalGenerators)
message(STATUS "  [2/6] SignalGenerators")

# 3. DataContext (depends: Interface)
add_subdirectory(DataContext)
message(STATUS "  [3/6] DataContext")

# 4. ModelsFunction (depends: Interface)
add_subdirectory(ModelsFunction)
message(STATUS "  [4/6] ModelsFunction")

# 5. Tester (depends: Interface, DataContext)
add_subdirectory(Tester)
message(STATUS "  [5/6] Tester")

# 6. MainProgram (depends: ALL)
add_subdirectory(MainProgram)
message(STATUS "  [6/6] MainProgram")

message(STATUS "=== CudaCalc Modules Complete ===")

# ===================================================================
# INSTALL (optional)
# ===================================================================
install(TARGETS cudacalc_fft16_test DESTINATION bin)

# ===================================================================
# SUMMARY
# ===================================================================
message(STATUS "")
message(STATUS "===================================================")
message(STATUS "CudaCalc Configuration Summary")
message(STATUS "===================================================")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard:     C++${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA version:      ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA arch:         sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Compiler:")
message(STATUS "    - CXX:           ${CMAKE_CXX_COMPILER}")
message(STATUS "    - CUDA:          ${CMAKE_CUDA_COMPILER}")
message(STATUS "  Output dir:        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "===================================================")
message(STATUS "")

