#include <cmath>
#include <iostream>
#include <chrono>

const int N = 1000010;
const size_t size = N * sizeof(float);

void vec_add(float* A, float* B, float* C)
{
    for(int i = 0; i < N; i ++)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void vec_add_cu(float* A, float* B, float* C, int N)
{
    int i = threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    // generate array:
    float* host_a = new float[N];
    float* host_b = new float[N];
    float* host_c = new float[N];
    float* valid_result = new float[N];

    // allocate cuda memory.
    float* device_a;
    cudaMalloc(&device_a, size);
    float* device_b;
    cudaMalloc(&device_b, size);
    float* device_c;
    cudaMalloc(&device_c, size);

    for(int i = 0; i < N; i ++)
    {
        host_a[i] = ((float) rand() / (RAND_MAX));
        host_b[i] = ((float) rand() / (RAND_MAX));
        host_c[i] = ((float) rand() / (RAND_MAX));
    }
    
    auto start = std::chrono::system_clock::now();

    vec_add(host_a, host_b, host_c);

    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;
    std::cout << "No cuda: " << elapsed.count() << '\n';

    std::cout << "[INFO]: Start accelerate compute with cuda.\n";
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, host_c, size, cudaMemcpyHostToDevice);
    
    start = std::chrono::system_clock::now();
    // must be memory allocated by cudaAllocate
    int threadPerBlock = 256;
    int blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;

    vec_add_cu<<<blockPerGrid, threadPerBlock>>>(device_a, device_b, device_c, N);

    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "With cuda: " << elapsed.count() << '\n';

    cudaMemcpy(valid_result, device_c, size, cudaMemcpyDeviceToHost);
    
    std::cout << "[INFO] Start validate with origin.\n";
    size_t error_num = 0;
    for(int i = 0; i < N; i ++)
    {
        if(std::abs(host_c[i] - valid_result[i]) > 1e-5)
        {
            std::cout << "Error in #" << i << "\n"; 
            error_num ++;
        }
    }

    std::cout << "Error result: " << error_num << std::endl;

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);
    free(valid_result);

    return 0;
}