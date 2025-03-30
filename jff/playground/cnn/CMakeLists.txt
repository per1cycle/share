cmake_minimum_required(VERSION 3.6)

project(cnn)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-lz")

add_executable(cnn cnn.cc)
