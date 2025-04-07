cmake_minimum_required(VERSION 3.10)
project(clfun)

# Set the C++ standard (optional, but good practice)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# Find the OpenCL framework on macOS
find_package(OpenCL REQUIRED)

function(add_cl_exe TARGET SOURCE)
    add_executable(${TARGET} ${SOURCE})
    target_link_libraries(${TARGET} OpenCL::OpenCL)
endfunction(add_cl_exe)

add_subdirectory(SGEMM)
add_subdirectory(book)
add_subdirectory(skeleton)