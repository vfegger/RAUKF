cmake_minimum_required(VERSION 3.24.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61 86)

project(HFE_RAUKF VERSION 0.1)

enable_language(CUDA)

# Add Libraries
add_subdirectory(src/structure)
add_subdirectory(src/hfe)
add_subdirectory(src/hfe2D)
add_subdirectory(src/math)
add_subdirectory(src/model)
add_subdirectory(src/timer)
add_subdirectory(src/filter)
add_subdirectory(src/statistics)

# Add executables
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC Structure HFE Math Model Timer Filter Statistics)

add_executable(${PROJECT_NAME}_test main_test.cpp)
target_link_libraries(${PROJECT_NAME}_test PUBLIC Structure HFE2D Math Model Timer Filter Statistics)