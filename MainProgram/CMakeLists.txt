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

message(STATUS "  MainProgram: Executables configured (FFT16, FFT32)")

