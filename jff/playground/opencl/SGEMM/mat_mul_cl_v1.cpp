#include <iostream>
#include <fstream>
#include <sstream>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif
void usage()
{
    std::cerr << "Usage: ./vX <path to cl>" << std::endl;
}

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
        usage();
        exit(1);
    }

    // assuming Current directory is build/
    std::cout << "[===] Running v1 optimize for matmul." << std::endl;
    std::string kernel_path(argv[1]);
    std::string kernel_code = load_kernel_code(kernel_path);
    std::cout << "Kernel code: " << std::endl;
    std::cout << kernel_code << std::endl;

    cl_platform_id platform_id;
    cl_int tmp;
    cl_uint platform_nums;
    clGetPlatformIDs(tmp, &platform_id, &platform_nums);
    std::cout << platform_nums << std::endl;
    return 0;
}