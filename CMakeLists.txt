cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_KERNELS CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-O3 --use_fast_math -Wno-deprecated-gpu-targets")

include_directories(./include)

add_subdirectory(kernels/softmax)
add_subdirectory(kernels/relu)
add_subdirectory(kernels/gemm)
add_subdirectory(kernels/layernorm)
add_subdirectory(kernels/gelu)
add_subdirectory(kernels/attention)