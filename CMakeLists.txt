cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cuckoo-hash VERSION 0.1.0)

# Enable compilation database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set output directories for all builds (Debug, Release, etc.)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)  # For executables
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)  # For shared libraries (.so, .dll)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)  # For static libraries (.a, .lib)

# Ensure the paths are set for multi-configuration generators (like Visual Studio)
if(CMAKE_CONFIGURATION_TYPES)
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
    endforeach()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add compilation flag(s) globally
add_compile_options(-march=native)

add_executable(cuckoo-hash-test tests/main.cpp)
target_include_directories(cuckoo-hash-test PRIVATE ${CMAKE_SOURCE_DIR}/src)
