cmake_minimum_required(VERSION 3.31)

project(MallatsAlgo LANGUAGES CXX CUDA)

add_compile_options(-Wno-deprecated-gpu-targets)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("----- Using DEBUG version -----")
    add_compile_options(-g -G)
    add_executable(mallat_tests)
    target_sources(mallat_tests PRIVATE tests.cu)
endif()

add_executable(mallat)
target_sources(mallat PRIVATE example.cu)
