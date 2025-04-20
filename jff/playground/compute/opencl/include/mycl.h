#ifndef MYCL_H
#define MYCL_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif

#include "common.h"

/** 
 * wrapper for my cl demo usage.
 */
namespace mycl
{
/**
 * @return the platform numbers, can be used by select_device()
 */
cl_uint get_platform_num()
{
    cl_uint platform_nums;
    cl_int status;
    status = clGetPlatformIDs(0, NULL, &platform_nums);
    CL_CHECK(status);

    return platform_nums;
}

cl_platform_id select_platform(int std_platform_id)
{
    cl_uint platform_numbers = get_platform_num();
    cl_int status;
    cl_platform_id *platforms = new cl_platform_id[platform_numbers];
    status = clGetPlatformIDs(platform_numbers, platforms, NULL);
    CL_CHECK(status);

    return platforms[std_platform_id];
}

cl_uint get_device_num_of_platform(int std_platform_id)
{
    cl_platform_id platform_id = select_platform(std_platform_id);
    cl_uint num_of_device_on_current_platform;
    cl_int status;
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_of_device_on_current_platform);
    CL_CHECK(status);

    return num_of_device_on_current_platform;
}

cl_device_id select_device(int std_platform_id, int std_device_id)
{
    cl_platform_id platform_id = select_platform(std_platform_id);
    // list devices of the platform
    cl_uint nums_of_device_on_current_platform = get_device_num_of_platform(std_platform_id);
    cl_int status;
    cl_device_id *devices = new cl_device_id[nums_of_device_on_current_platform];

    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, nums_of_device_on_current_platform, devices, NULL);
    CL_CHECK(status);

    return devices[std_device_id];
}

void list_platform_and_device()
{
    cl_uint num_of_platform = get_platform_num();

    for(int i = 0; i < num_of_platform; i ++)
    {
        std::cout << "Platform; " << select_platform(i) << std::endl;
        cl_uint num_of_device = get_device_num_of_platform(i);
        for(int j = 0; j < num_of_device; j ++)
        {
            std::cout << '\t';
            std::cout << select_device(i, j);
            std::cout << std::endl;
        } 
    }
}

// used for histagram.
std::vector<int> generate_4x4_image()
{
    
}

std::vector<int> 

} // namespace mycl
#endif // MYCL_H
