#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif

#define CL_CHECK(err) do \
{ \
    if(err) { \
        std::cout << "CL_CHECK ERROR: code <" << err << ">" << std::endl;   \
    } \
} while(0)

int main()
{
    // need call clGetPlatformIDs twice, first time got the num platforms.
    cl_int status;
    cl_uint num_platforms = 0;
    cl_uint num_device_per_platform = 0;
    status = clGetPlatformIDs(0, NULL, &num_platforms);
    CL_CHECK(status);
    
    cl_platform_id *platforms = new cl_platform_id[num_platforms];
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    CL_CHECK(status);
    
    // get platforms and device info.
    for(cl_uint i = 0; i < num_platforms; i ++)
    {
        std::cout << "Platform: " << platforms[i] << " has devices:" << std::endl;
        // device id also need be call twice 
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_device_per_platform);

        cl_device_id *devices = new cl_device_id[num_device_per_platform];
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_device_per_platform, devices, NULL);
        CL_CHECK(status);

        for(cl_int j = 0; j < num_device_per_platform; j ++)
        {
            std::cout << '\t' << devices[j] << std::endl;
        }
    }

    cl_context context;
    cl_device_id device_id;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_device_per_platform, &device_id, NULL);
    context = clCreateContext(NULL, num_device_per_platform, &device_id, NULL, NULL, &status);
    CL_CHECK(status);

    // create command queue for platform.
    cl_command_queue command_queue;
    command_queue = clCreateCommandQueue(context, device_id, 0, &status);
    CL_CHECK(status);

    // allocate data to process.
    cl_mem arr_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
    cl_mem arr_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &status);
    cl_mem arr_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &status);

    
    return 0;
}