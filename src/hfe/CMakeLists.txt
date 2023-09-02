cmake_minimum_required(VERSION 3.24.0)

set(target HFE)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE source ${source_path}/*.cpp ${source_path}/*.cu)

find_package(CUDAToolkit)

add_library(${target} ${source})

target_include_directories(${target} PUBLIC ${include_path})
target_link_libraries(${target} PUBLIC Structure PUBLIC CUDA::cudart PUBLIC CUDA::cublas PUBLIC CUDA::cusolver PUBLIC CUDA::curand)
