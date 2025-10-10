# SignalGenerators Module - Test signal generation

add_library(SignalGenerators STATIC)

target_sources(SignalGenerators
    PRIVATE
        src/placeholder.cpp
        # Will add: src/sine_generator.cpp
)

target_include_directories(SignalGenerators
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(SignalGenerators
    PUBLIC
        Interface
)

message(STATUS "  SignalGenerators: Static library configured")

