#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

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
        std::cout << "Gpu name: " << prop.name << std::endl;
    }

    return 0;
}
