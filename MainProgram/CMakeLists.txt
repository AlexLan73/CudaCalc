# MainProgram Module - Entry point for FFT16 testing

add_executable(cudacalc_fft16_test)

target_sources(cudacalc_fft16_test
    PRIVATE
        # Will add: src/main_fft16_test.cpp
        src/main.cpp  # Temporary placeholder
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

message(STATUS "  MainProgram: Executable configured")

