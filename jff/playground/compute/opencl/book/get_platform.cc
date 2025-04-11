#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif

#define CL_CHECK(err) do { \
    if(err) { \
        std::cout << "CL_CHECK ERROR: code <" << err << ">" << std::endl;   \
    } \
} while(0)

int main()
{
    cl_int cl_status;
    cl_platform_id platform_id;
    cl_uint num_platforms;
    cl_status = clGetPlatformIDs(1, &platform_id, &num_platforms);
    CL_CHECK(cl_status);

    std::cout << "num of platforms: " << num_platforms << std::endl;
    cl_device_id device_id;
    cl_status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    CL_CHECK(cl_status);
 
    return 0;
}