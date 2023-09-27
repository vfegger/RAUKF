cmake_minimum_required(VERSION 3.24.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61 86)

project(HFE_RAUKF VERSION 0.1)

# Add Libraries
add_subdirectory(src/structure)
add_subdirectory(src/hfe)
add_subdirectory(src/math)
add_subdirectory(src/model)
add_subdirectory(src/timer)
add_subdirectory(src/filter)
add_subdirectory(src/statistics)

# Add executables
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC Structure HFE Math Model Timer Filter Statistics)