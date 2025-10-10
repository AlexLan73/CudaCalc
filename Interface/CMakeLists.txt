# Interface Module - Header-only base interfaces and data structures

add_library(Interface INTERFACE)

target_include_directories(Interface
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

message(STATUS "  Interface: Header-only library configured")

