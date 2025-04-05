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

void print_arr(int *arr, int arr_size)
{
    for(int i = 0; i < arr_size; i ++)
    {
        std::cout << arr[i] << ' ';
        if(i && i % 64 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::string load_kernel_code(const std::string& kernel_path)
{
    std::cout << kernel_path << std::endl;
    std::ifstream f(kernel_path.c_str());
    std::stringstream buf;
    buf << f.rdbuf();

    return buf.str();
}

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

    // allocate data to process
    const int ARRAY_SIZE = 64;
    int *h_a = new int[ARRAY_SIZE];
    int *h_b = new int[ARRAY_SIZE];
    int *h_c = new int[ARRAY_SIZE];

    for(int i = 0; i < ARRAY_SIZE; i ++)
    {
        h_a[i] = i;
        h_b[i] = i;
        h_c[i] = 0;
    }

    // print_arr(h_a, ARRAY_SIZE);
    // print_arr(h_b, ARRAY_SIZE);

    // method 1
    // 
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * ARRAY_SIZE, NULL, &status);
    CL_CHECK(status);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * ARRAY_SIZE, NULL, &status);
    CL_CHECK(status);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * ARRAY_SIZE, NULL, &status);
    CL_CHECK(status);

    status = clEnqueueWriteBuffer(command_queue, d_a, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, h_a, 0, NULL, NULL);
    CL_CHECK(status);
    status = clEnqueueWriteBuffer(command_queue, d_b, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, h_b, 0, NULL, NULL);
    CL_CHECK(status);
    // shouldn't be called.
    status = clEnqueueWriteBuffer(command_queue, d_c, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, h_c, 0, NULL, NULL);
    CL_CHECK(status);

//     const char *kernel_code = 
// "__kernel void add(__global int *A, __global int *B, __global int *C)"\
// "{" \
// "    int thread_id = get_global_id(0);" \
// "    C[thread_id] = A[thread_id] + B[thread_id];" \
// "}";
    std::string kernel_source = load_kernel_code("/Users/z/Projects/Github/dev/share/jff/playground/opencl/book/add.cl");
    char *kernel_code = new char[kernel_source.size() + 1];
    strcpy(kernel_code, kernel_source.c_str());

    std::cout << kernel_code << std::endl;
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, NULL, &status);
    CL_CHECK(status);
    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    CL_CHECK(status);

    // start run kernel.
    cl_kernel kernel = clCreateKernel(program, "add", &status);
    CL_CHECK(status);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    CL_CHECK(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    CL_CHECK(status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    CL_CHECK(status);

    size_t index_space_size[1], work_group_size[1];
    index_space_size[0] = ARRAY_SIZE;
    work_group_size[0] = 64;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, index_space_size, work_group_size, 0, NULL, NULL);
    status = clEnqueueReadBuffer(command_queue, d_c, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, h_c, 0, NULL, NULL);
    print_arr(h_c, ARRAY_SIZE);
    return 0;
}