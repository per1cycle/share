#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

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
    clGetPlatformIDs(1, NULL, &platform_nums);

    // setup matrix
    const int N = 10, K = 10, M = 5;

    float* A = new float[N * K];
    float* B = new float[K * M];
    float* C = new float[N * M];
    for(int i = 0; i < N * K; ++i)
        A[i] = static_cast<float>(i);
    for(int i = 0; i < K * M; ++i)
        B[i] = static_cast<float>(i);

    for(int i = 0; i < N * M; ++i)
        C[i] = 0.0f;
    
    // setup opencl 
    clGetPlatformIDs(1, &platform_id, &platform_nums);
    cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
    cl_context context = clCreateContext(properties, 0, NULL, NULL, NULL, &tmp);
    cl_command_queue queue = clCreateCommandQueue(context, 0, 0, &tmp);
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * K, A, &tmp);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * K * M, B, &tmp);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * M, NULL, &tmp);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)kernel_code.c_str(), NULL, &tmp);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "sgemm", &tmp);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(int), &K);
    clSetKernelArg(kernel, 5, sizeof(int), &M);
    size_t global_work_size[2] = {N, M};
    size_t local_work_size[2] = {16, 16};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, sizeof(float) * N * M, C, 0, NULL, NULL);
    clFinish(queue);
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    // copy result to c

    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < M; ++j)
        {
            std::cout << std::setw(5) << std::setprecision(5) << C[i * M + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}