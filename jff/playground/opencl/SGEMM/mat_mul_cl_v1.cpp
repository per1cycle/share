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

void simple_matmul(float *a, float *b, float *c, int N, int K, int M)
{
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            float temp = 0.0f;
            for(int k = 0; k < K; k ++)
            {
                temp += a[k + i * K] * b[k * M + j];
            }
            c[i * M + j] = temp;
        }
    }
}

void print_2d(float *a, int row, int col)
{
    for(int i = 0; i < 10; i ++) 
        std::cout << '+';
    std::cout << std::endl;
    std::cout << "[";
    for(int i = 0; i < row; i ++)
    {
        std::cout << '[';
        for(int j = 0; j < col; j ++)
        {
            std::cout << a[i * col + j] << ", ";
        }
        if(i != row - 1)
            std::cout << "]," << std::endl;
        else 
            std::cout << "]";
    }
    std::cout << "]" << std::endl;
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
    // std::cout << str << std::endl;

    cl_device_id device_id;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_device_per_platform, &device_id, NULL);
    CL_CHECK(status);

    const int N = 256, M = 256, K = 128;
    float *h_a = new float[N * K];
    float *h_b = new float[K * M];
    float *h_c = new float[N * M];

    // initialize the two matrix
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < K; j ++)
        {
            h_a[i * K + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    for(int i = 0; i < K; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            h_b[i * M + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    cl_context context;
    context = clCreateContext(NULL, num_device_per_platform, &device_id, NULL, NULL, &status);
    CL_CHECK(status);

    cl_command_queue command_queue;
    command_queue = clCreateCommandQueue(context, device_id, 0, &status);
    CL_CHECK(status);

    print_2d(h_a, N, K);
    print_2d(h_b, K, M);
    simple_matmul(h_a, h_b, h_c, N, K, M);
    print_2d(h_c, N, M);
    return 0;
}