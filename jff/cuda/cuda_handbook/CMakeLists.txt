cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(cuda_handbook)

find_package(CUDA REQUIRED LANGUAGES C CXX CUDA)

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/clang; 
  -gencode;arch=compute_30,code=sm_30;  
  -gencode;arch=compute_35,code=sm_35;
<<<<<<< HEAD
  -gencode;arch=compute_36,code=compute_35;
=======
  -gencode;arch=compute_35,code=compute_35;
>>>>>>> e0a1b62 (Upload submodule and init the handbook project.)
  -gencode;arch=compute_20,code=sm_20; 
  -gencode;arch=compute_11,code=sm_11; 
  -gencode;arch=compute_12,code=sm_12;
  -gencode;arch=compute_13,code=sm_13;")
  # my gpu is 1060
  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
  set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 
<<<<<<< HEAD
        # 100 101 120
        )
  add_subdirectory(playground)
  add_subdirectory(ch3)
=======
        # 100 101 120 not support for 1060
        )
  add_subdirectory(playground)
>>>>>>> e0a1b62 (Upload submodule and init the handbook project.)

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
