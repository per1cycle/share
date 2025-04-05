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

std::string load_kernel_code(const std::string& kernel_path)
{
    std::cout << kernel_path << std::endl;
    std::ifstream f(kernel_path.c_str());
    std::stringstream buf;
    buf << f.rdbuf();

    return buf.str();
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        exit(1);
    }
    // need call clGetPlatformIDs twice, first time got the num platforms.
    cl_int status;
    cl_uint num_platforms = 0;
    cl_uint num_device_per_platform = 0;
    status = clGetPlatformIDs(0, NULL, &num_platforms);
    CL_CHECK(status);
    
    cl_platform_id *platforms = new cl_platform_id[num_platforms];
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    CL_CHECK(status);

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

    std::string kernel_source = load_kernel_code(argv[1]);
    char* str = new char[kernel_source.size() + 1];
    std::strcpy(str, kernel_source.c_str());

    return 0;
}