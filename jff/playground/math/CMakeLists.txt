cmake_minimum_required(VERSION 3.10)
project(math)

# Set the C++ standard (optional, but good practice)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fft fft.cc)