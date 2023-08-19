cmake_minimum_required(VERSION 3.24.0)

set(target Math)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE source ${source_path}/*.cpp ${source_path}/*.cu)

find_package(CUDAToolkit)

add_library(${target} ${source})

target_include_directories(${target} PUBLIC ${include_path})
target_link_libraries(${target} PUBLIC Structure PRIVATE CUDA::cudart PRIVATE CUDA::cublas PRIVATE CUDA::cusolver)

#find_package(OpenMP REQUIRED)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
#endif()