cmake_minimum_required(VERSION 3.10)
project(clfun)

# Set the C++ standard (optional, but good practice)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
# Find the OpenCL framework on macOS

function(add_cl_exe TARGET SOURCE)
    add_executable(${TARGET} ${SOURCE})
    target_link_libraries(${TARGET} OpenCL::OpenCL ${OpenCV_LIBS})
endfunction(add_cl_exe)

add_subdirectory(SGEMM)
add_subdirectory(book)
add_subdirectory(img)
add_subdirectory(skeleton)