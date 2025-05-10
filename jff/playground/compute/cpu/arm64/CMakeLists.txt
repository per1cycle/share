cmake_minimum_required(VERSION 2.8)

project(sgemm_opt_cpu)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(main main.cc)
