cmake_minimum_required(VERSION 3.5)

project(sgemm_opt_cpu)

include_directories(${CMAKE_SOURCE_DIR})
add_compile_options(-mavx512f)

add_executable(main main.cc)
