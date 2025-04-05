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
    char* kernel_code = new char[kernel_source.size() + 1];
    std::strcpy(kernel_code, kernel_source.c_str());
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

    // print_2d(h_a, N, K);
    // print_2d(h_b, K, M);
    // simple_matmul(h_a, h_b, h_c, N, K, M);
    // print_2d(h_c, N, M);
    
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N * K, NULL, &status);
    CL_CHECK(status);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * K * M, NULL, &status);
    CL_CHECK(status);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * M, NULL, &status);
    CL_CHECK(status);

    // copy host mem to device
    status = clEnqueueWriteBuffer(command_queue, d_a, CL_TRUE, 0, sizeof(float) * N * K, h_a, 0, NULL, NULL);
    CL_CHECK(status);
    status = clEnqueueWriteBuffer(command_queue, d_b, CL_TRUE, 0, sizeof(float) * K * M, h_b, 0, NULL, NULL);
    CL_CHECK(status);
    status = clEnqueueWriteBuffer(command_queue, d_c, CL_TRUE, 0, sizeof(float) * N * M, h_c, 0, NULL, NULL);
    CL_CHECK(status);

    cl_program program;
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, NULL, &status);
    CL_CHECK(status);
    
    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    CL_CHECK(status);

    cl_kernel kernel;
    kernel = clCreateKernel(program, "v1", &status);
    CL_CHECK(status);

    status = clSetKernelArg(kernel, 0, sizeof(int), &N);
    CL_CHECK(status);
    status = clSetKernelArg(kernel, 1, sizeof(int), &M);
    CL_CHECK(status);
    status = clSetKernelArg(kernel, 2, sizeof(int), &K);
    CL_CHECK(status);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_a);
    CL_CHECK(status);
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_b);
    CL_CHECK(status);
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_c);
    CL_CHECK(status);

    return 0;
}