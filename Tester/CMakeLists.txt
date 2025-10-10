# Tester Module - Profiling and validation

add_library(Tester STATIC)

target_sources(Tester
    PRIVATE
        src/performance/basic_profiler.cpp
        src/validation/fft_validator.cpp
        # Will add:
        # src/performance/memory_profiler.cpp
)

target_include_directories(Tester
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(Tester
    PUBLIC
        Interface
        DataContext
        CUDA::cudart
        CUDA::cufft
)

message(STATUS "  Tester: Static library configured")

