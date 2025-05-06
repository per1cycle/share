cmake_minimum_required(VERSION 3.15) # HIP language support requires 3.21
project(sgemm_hip )
set(CMAKE_CXX_STANDARD 17)

// set compiler to hipcc
set(CMAKE_CXX_COMPILER hipcc)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sgemm_skeleton sgemm_skeleton.cc)
