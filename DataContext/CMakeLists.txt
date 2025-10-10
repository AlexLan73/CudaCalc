# DataContext Module - Data management, logging, archiving

add_library(DataContext STATIC)

target_sources(DataContext
    PRIVATE
        src/json_logger.cpp
        # Will add: src/model_archiver.cpp
)

target_include_directories(DataContext
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(DataContext
    PUBLIC
        Interface
        nlohmann_json::nlohmann_json
)

message(STATUS "  DataContext: Static library configured")

