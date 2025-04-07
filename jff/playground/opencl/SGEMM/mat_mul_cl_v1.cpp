#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif

#define CL_CHECK(err) do \
{ \
    if(err) { \
        std::cout << "CL_CHECK ERROR: code <" << err << ">: "; \
        switch(err) \
        { \
            case -11: \
                std::cout << "BUILD_PROGRAM_FAILURE"; \
                break; \
            case -3: \
                std::cout << "COMPILER_NOT_AVAILABLE"; \
                break; \
            case -2: \
                std::cout << "DEVICE_NOT_AVAILABLE"; \
                break; \
            case -1: \
                std::cout << "DEVICE_NOT_FOUND"; \
                break; \
            case -14: \
                std::cout << "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; \
                break; \
            case -9: \
                std::cout << "IMAGE_FORMAT_MISMATCH"; \
                break; \
            case -10: \
                std::cout << "IMAGE_FORMAT_NOT_SUPPORTED"; \
                break; \
            case -49: \
                std::cout << "INVALID_ARG_INDEX"; \
                break; \
            case -51: \
                std::cout << "INVALID_ARG_SIZE"; \
                break; \
            case -50: \
                std::cout << "INVALID_ARG_VALUE"; \
                break; \
            case -42: \
                std::cout << "INVALID_BINARY"; \
                break; \
            case -61: \
                std::cout << "INVALID_BUFFER_SIZE"; \
                break; \
            case -43: \
                std::cout << "INVALID_BUILD_OPTIONS"; \
                break; \
            case -36: \
                std::cout << "INVALID_COMMAND_QUEUE"; \
                break; \
            case -34: \
                std::cout << "INVALID_CONTEXT"; \
                break; \
            case -33: \
                std::cout << "INVALID_DEVICE"; \
                break; \
            case -31: \
                std::cout << "INVALID_DEVICE_TYPE"; \
                break; \
            case -58: \
                std::cout << "INVALID_EVENT"; \
                break; \
            case -57: \
                std::cout << "INVALID_EVENT_WAIT_LIST"; \
                break; \
            case -56: \
                std::cout << "INVALID_GLOBAL_OFFSET"; \
                break; \
            case -63: \
                std::cout << "INVALID_GLOBAL_WORK_SIZE"; \
                break; \
            case -60: \
                std::cout << "INVALID_GL_OBJECT"; \
                break; \
            case -37: \
                std::cout << "INVALID_HOST_PTR"; \
                break; \
            case -39: \
                std::cout << "INVALID_IMAGE_FORMAT_DESCRIPTOR"; \
                break; \
            case -40: \
                std::cout << "INVALID_IMAGE_SIZE"; \
                break; \
            case -48: \
                std::cout << "INVALID_KERNEL"; \
                break; \
            case -52: \
                std::cout << "INVALID_KERNEL_ARGS"; \
                break; \
            case -47: \
                std::cout << "INVALID_KERNEL_DEFINITION"; \
                break; \
            case -46: \
                std::cout << "INVALID_KERNEL_NAME"; \
                break; \
            case -38: \
                std::cout << "INVALID_MEM_OBJECT"; \
                break; \
            case -62: \
                std::cout << "INVALID_MIP_LEVEL"; \
                break; \
            case -59: \
                std::cout << "INVALID_OPERATION"; \
                break; \
            case -32: \
                std::cout << "INVALID_PLATFORM"; \
                break; \
            case -44: \
                std::cout << "INVALID_PROGRAM"; \
                break; \
            case -45: \
                std::cout << "INVALID_PROGRAM_EXECUTABLE"; \
                break; \
            case -64: \
                std::cout << "INVALID_PROPERTY"; \
                break; \
            case -35: \
                std::cout << "INVALID_QUEUE_PROPERTIES"; \
                break; \
            case -41: \
                std::cout << "INVALID_SAMPLER"; \
                break; \
            case -30: \
                std::cout << "INVALID_VALUE"; \
                break; \
            case -53: \
                std::cout << "INVALID_WORK_DIMENSION"; \
                break; \
            case -54: \
                std::cout << "INVALID_WORK_GROUP_SIZE"; \
                break; \
            case -55: \
                std::cout << "INVALID_WORK_ITEM_SIZE"; \
                break; \
            case -12: \
                std::cout << "MAP_FAILURE"; \
                break; \
            case -8: \
                std::cout << "MEM_COPY_OVERLAP"; \
                break; \
            case -4: \
                std::cout << "MEM_OBJECT_ALLOCATION_FAILURE"; \
                break; \
            case -13: \
                std::cout << "MISALIGNED_SUB_BUFFER_OFFSET"; \
                break; \
            case -6: \
                std::cout << "OUT_OF_HOST_MEMORY"; \
                break; \
            case -5: \
                std::cout << "OUT_OF_RESOURCES"; \
                break; \
            case -7: \
                std::cout << "PROFILING_INFO_NOT_AVAILABLE"; \
                break; \
            case 0: \
                std::cout << "SUCCESS"; \
                break; \
            default: \
                std::cout << "Unknown error code"; \
        } \
        std::cout << std::endl; \
    } \
} while(0)

void usage()
{
    std::cerr << "Usage: ./vX <path to cl>" << std::endl;
}

std::string load_kernel_code(const std::string& kernel_path)
{
    // std::cout << kernel_path << std::endl;
    std::ifstream f(kernel_path.c_str());
    std::stringstream buf;
    buf << f.rdbuf();
    return buf.str();
}

void load_numpy()
{
    std::cout << "import numpy as np" << std::endl;
}

void numpy_dot(std::string arr_a, std::string arr_b, std::string arr_compare)
{
    std::cout << "tmp = ";
    std::cout << arr_a << ".dot(" << arr_b << ")" << std::endl;
    std::cout << "res = np.allclose(" << "tmp" << ", " << arr_compare << ")" << std::endl;
    std::cout << "print(res)" << std::endl;
}

void gen_info(float *a, float *b, float *c, int N, int M, int K, std::uint32_t flags)
{

}

void gen_py(float *A, float *B, float *C, int N, int M, int K)
{

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

void print_2d(float *a, int row, int col, std::string arr_name)
{
    std::cout << arr_name << " = np.array(";
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
    std::cout << "])" << std::endl;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        exit(1);
    }
    // load_numpy();
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
        // std::cout << "Platform: " << platforms[i] << " has devices:" << std::endl;
        // device id also need be call twice 
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_device_per_platform);

        cl_device_id *devices = new cl_device_id[num_device_per_platform];
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_device_per_platform, devices, NULL);
        CL_CHECK(status);

        // for(cl_int j = 0; j < num_device_per_platform; j ++)
        // {
        //     std::cout << '\t' << devices[j] << std::endl;
        // }
    }

    std::string kernel_source = load_kernel_code(argv[1]);
    char* kernel_code = new char[kernel_source.size() + 1];
    std::strcpy(kernel_code, kernel_source.c_str());
    // std::cout << str << std::endl;

    cl_device_id device_id;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_device_per_platform, &device_id, NULL);
    CL_CHECK(status);

    // const int N = 8192, M = 8192, K = 8192;
    const int N = 8192, M = 4096, K = 4096;
    // const int N = 16, M = 16, K = 16;
    float *h_a = new float[N * K];
    float *h_b = new float[K * M];
    float *h_c = new float[N * M];
    float *result_from_host = new float[N * M];

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

    // print_2d(h_a, N, K, "h_a");
    // print_2d(h_b, K, M, "h_b");
    // simple_matmul(h_a, h_b, result_from_host, N, K, M);
    // print_2d(h_c, N, M);
    
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N * K, NULL, &status);
    CL_CHECK(status);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * K * M, NULL, &status);
    CL_CHECK(status);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * M, NULL, &status);
    CL_CHECK(status);

    auto start = std::chrono::high_resolution_clock::now();
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
    kernel = clCreateKernel(program, "v2", &status);
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

    const int TS = 16;
    size_t g_work[2] = {N, M};
    size_t l_work[2] = {TS, TS};

    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, g_work, l_work, 0, NULL, NULL);
    CL_CHECK(status);
    // measure time
    // simple_matmul(h_a, h_b, result_from_host, N, K, M);
    clEnqueueReadBuffer(command_queue, d_c, CL_TRUE, 0, sizeof(float) * N * M, h_c, 0, NULL, NULL);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    
    std::cout << "Flops: " << std::setw(5) << std::setprecision(5) << 2.0f * N * M * K / elapsed.count() * 1000 / 1000 / 1000 / 1000 / 1000 << " TFLOPS." << std::endl;

    // print_2d(h_c, N, M, "h_c");
    // numpy_dot("h_a", "h_b", "h_c");

    return 0;
}