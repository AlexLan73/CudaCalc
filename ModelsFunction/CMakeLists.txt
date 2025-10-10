# ModelsFunction Module - GPU implementations (FFT kernels)

add_library(ModelsFunction STATIC)

target_sources(ModelsFunction
    PRIVATE
        src/nvidia/fft/FFT16_Shared2D/fft16_shared2d.cpp
        src/nvidia/fft/FFT16_Shared2D/fft16_shared2d_kernel.cu
        src/nvidia/fft/FFT16_Shared2D/fft16_shared2d_profiled.cpp
        src/nvidia/fft/FFT16_WMMA/fft16_wmma.cpp
        src/nvidia/fft/FFT16_WMMA/fft16_wmma_kernel.cu
        src/nvidia/fft/FFT16_WMMA/fft16_wmma_profiled.cpp
        src/nvidia/fft/FFT16_WMMA/fft16_wmma_optimized.cpp
        src/nvidia/fft/FFT16_WMMA/fft16_wmma_optimized_kernel.cu
        src/nvidia/fft/FFT16_WMMA/fft16_wmma_optimized_profiled.cpp
        src/nvidia/fft/FFT16_WMMA_Ultra/fft16_wmma_ultra.cpp
        src/nvidia/fft/FFT16_WMMA_Ultra/fft16_wmma_ultra_kernel.cu
        src/nvidia/fft/FFT16_WMMA_Ultra/fft16_wmma_ultra_profiled.cpp
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_optimized.cpp
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_optimized_kernel.cu
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_optimized_profiled.cpp
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_v2_ultra.cpp
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_v2_ultra_kernel.cu
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_v3_final.cpp
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_v3_final_kernel.cu
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_v4_correct.cpp
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_v4_correct_kernel.cu
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_v5_fixed.cpp
        src/nvidia/fft/FFT32_WMMA/fft32_wmma_v5_fixed_kernel.cu
)

target_include_directories(ModelsFunction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(ModelsFunction
    PUBLIC
        Interface
        CUDA::cufft
        CUDA::cudart
)

# CUDA specific settings
set_target_properties(ModelsFunction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

message(STATUS "  ModelsFunction: CUDA library configured (sm_86)")

