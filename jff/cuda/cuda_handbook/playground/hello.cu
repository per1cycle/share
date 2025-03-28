<<<<<<< HEAD
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

__device__ void cuda_hello()
{
}

int main()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "GPUs: " << device_count << std::endl;

    for(int i = 0; i < device_count; i ++)
    {
        int dev;
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "Gpu name:" << prop.name << std::endl;
    }
=======
#include "cuda_runtime.h"
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
>>>>>>> e0a1b62 (Upload submodule and init the handbook project.)
    return 0;
}
