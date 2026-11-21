cmake_minimum_required(VERSION 3.18)
project(cuda_batch_image_processor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "Building with genuine CUDA GPU acceleration support.")
else()
    message(WARNING "No CUDA compiler found. Building with host CPU reference & validation mode.")
endif()

include_directories(include)

set(SOURCES
    src/main.cpp
    src/image_processor.cpp
    src/image_io.cpp
    src/cpu_reference.cpp
    src/benchmark.cpp
    src/cuda_kernels.cu
)

if(CMAKE_CUDA_COMPILER)
    add_executable(image_processor ${SOURCES})
    set_target_properties(image_processor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(image_processor PRIVATE USE_CUDA=1)
else()
    set_source_files_properties(src/cuda_kernels.cu PROPERTIES LANGUAGE CXX)
    add_executable(image_processor ${SOURCES})
endif()

# Installation
install(TARGETS image_processor DESTINATION bin)
