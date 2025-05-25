cmake_minimum_required(VERSION 4.0)

project(cpp-tmp)
set(CMAKE_CXX_STANDARD 17)
include_directories(
    ${CMAKE_SOURCE_DIR}
)
add_compile_options(-save-temps -g)

add_subdirectory(playground)
add_subdirectory(modern-cpp-design)
add_subdirectory(cpp-template)
add_subdirectory(cpp-template-metaprogramming)