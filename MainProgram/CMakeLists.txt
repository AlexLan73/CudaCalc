# MainProgram Module - Entry point for FFT testing

# FFT16 Test
add_executable(cudacalc_fft16_test)

target_sources(cudacalc_fft16_test
    PRIVATE
        src/main.cpp
)

target_link_libraries(cudacalc_fft16_test
    PRIVATE
        Interface
        SignalGenerators
        DataContext
        ModelsFunction
        Tester
        CUDA::cudart
        CUDA::cufft
        nlohmann_json::nlohmann_json
)

# FFT32 Test
add_executable(cudacalc_fft32_test)

target_sources(cudacalc_fft32_test
    PRIVATE
        src/main_fft32.cpp
)

target_link_libraries(cudacalc_fft32_test
    PRIVATE
        Interface
        SignalGenerators
        DataContext
        ModelsFunction
        Tester
        CUDA::cudart
        CUDA::cufft
        nlohmann_json::nlohmann_json
)

# FFT32 Comparison Table Test
add_executable(cudacalc_fft32_table)

target_sources(cudacalc_fft32_table
    PRIVATE
        src/main_fft32_comparison_table.cpp
)

target_link_libraries(cudacalc_fft32_table
    PRIVATE
        Interface
        SignalGenerators
        DataContext
        ModelsFunction
        Tester
        CUDA::cudart
        CUDA::cufft
        nlohmann_json::nlohmann_json
)

# FFT32 V4 Quick Test
add_executable(test_fft32_v4)

target_sources(test_fft32_v4
    PRIVATE
        src/test_fft32_v4.cpp
)

target_link_libraries(test_fft32_v4
    PRIVATE
        Interface
        SignalGenerators
        DataContext
        ModelsFunction
        Tester
        CUDA::cudart
        CUDA::cufft
        nlohmann_json::nlohmann_json
)

# FFT32 V5 Quick Test
add_executable(test_fft32_v5)

target_sources(test_fft32_v5
    PRIVATE
        src/test_fft32_v5.cpp
)

target_link_libraries(test_fft32_v5
    PRIVATE
        Interface
        SignalGenerators
        DataContext
        ModelsFunction
        Tester
        CUDA::cudart
        CUDA::cufft
        nlohmann_json::nlohmann_json
)

# FFT64 Quick Test
add_executable(test_fft64)

target_sources(test_fft64
    PRIVATE
        src/test_fft64.cpp
)

target_link_libraries(test_fft64
    PRIVATE
        Interface
        SignalGenerators
        DataContext
        ModelsFunction
        Tester
        CUDA::cudart
        CUDA::cufft
        nlohmann_json::nlohmann_json
)

message(STATUS "  MainProgram: Executables configured (FFT16, FFT32, FFT32_Table, Test_V4, Test_V5, Test_FFT64)")

