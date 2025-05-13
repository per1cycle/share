cmake_minimum_required(VERSION 3.5)

project(sgemm_opt_cpu)
add_compile_options(-save-temps)
include_directories(${CMAKE_SOURCE_DIR})

add_executable(main main.cc)
