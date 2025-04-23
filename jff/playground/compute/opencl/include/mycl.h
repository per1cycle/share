#ifndef MYCL_H
#define MYCL_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif

#include "common.h"
#include <iomanip>

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

namespace simple
{
/**
 * create context with select device.
 */
cl_context simple_create_context(cl_device_id *device, int std_platform_id)
{
    cl_int status;
    cl_context ctx = clCreateContext(NULL, get_device_num_of_platform(std_platform_id), device, NULL, NULL, &status);
    CL_CHECK(status);

    return ctx;
}

cl_command_queue simple_create_command_queue(cl_context ctx, cl_device_id device)
{
    cl_int status;
    cl_command_queue cq = clCreateCommandQueue(ctx, device, 0, &status);
    CL_CHECK(status);
    return cq;
}

cl_mem simple_create_ro_cl_buffer(cl_context ctx, size_t size)
{
    cl_int status;

    cl_mem mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY, size, NULL, &status);
    CL_CHECK(status);
    return mem;
}

cl_mem simple_create_wo_cl_buffer(cl_context ctx, size_t size)
{
    cl_int status;

    cl_mem mem = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size, NULL, &status);
    CL_CHECK(status);
    return mem;
}

}
// used for histagram.
/**
 * generate N rows and M columns matrix.
 */
void generate_NxM_mat(float *dst, int N, int M)
{
    if(N <= 0 || M <= 0)
    {
    }

    for(int i = 0; i < N; i ++) // row
    {
        // image i th row j column.
        for(int j = 0; j < M; j ++)
        {
            dst[i * M + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }
}

void print_NxM_mat(float *dst, int N, int M)
{
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            std::cout << std::setw(6) << std::setfill('0') << dst[i * M + j] << ' ';
        }
        std::cout << std::endl;
    }
}

} // namespace mycl
#endif // MYCL_H
