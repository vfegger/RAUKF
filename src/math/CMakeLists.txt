cmake_minimum_required(VERSION 3.24.0)

set(target Math)
set(target_exe Math_Test)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE source ${source_path}/*.cpp ${source_path}/*.cu)
list(FILTER source EXCLUDE REGEX ".*MathTest\\.cpp$")

find_package(CUDAToolkit)

add_library(${target} ${source})
add_executable(${target_exe} src/MathTest.cpp ${source})

target_include_directories(${target} PUBLIC ${include_path})
target_link_libraries(${target} PUBLIC Structure PRIVATE CUDA::cudart PRIVATE CUDA::cublas PRIVATE CUDA::cusolver)

target_include_directories(${target_exe} PUBLIC ${include_path})
target_link_libraries(${target_exe} PUBLIC Structure PRIVATE CUDA::cudart CUDA::cublas PRIVATE CUDA::cusolver)

#find_package(OpenMP REQUIRED)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
#endif()