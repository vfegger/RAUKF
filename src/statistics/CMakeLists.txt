cmake_minimum_required(VERSION 3.24.0)

set(target Statistics)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE source ${source_path}/*.cpp)

# Julia Include
if(NOT "$ENV{JULIA_PATH}")
    message(STATUS "Path: $ENV{JULIA_PATH}")
    message(STATUS "Julia Installation and Enviroment Variable Found")
    set(JULIA_INCLUDE_DIR "$ENV{JULIA_PATH}/include/julia")
    set(JULIA_LIB_DIR "$ENV{JULIA_PATH}/lib")
    find_file(JULIA_LIBS NAMES libjulia.dll.a libopenlibm.dll.a HINTS ${JULIA_LIB_DIR})
else()
    message(STATUS "Path: $ENV{JULIA_PATH}")
    message(FATAL_ERROR "Julia Installation and Enviroment Variable Not Found")
endif()

add_library(${target} ${source})

target_include_directories(${target} PUBLIC ${include_path} PRIVATE ${JULIA_INCLUDE_DIR})
target_link_libraries(${target} PRIVATE ${JULIA_LIBS})