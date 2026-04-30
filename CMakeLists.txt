cmake_minimum_required(VERSION 3.31)
project(MallatsAlgo LANGUAGES CUDA)

#set(ENV{NVCC_CCBIN} "/usr/bin/gcc-12")
add_compile_options(-g -G --gpu-architecture=sm_61 -Wno-deprecated-gpu-targets)

add_executable(mallat)
target_sources(mallat PRIVATE mallat.cu)
