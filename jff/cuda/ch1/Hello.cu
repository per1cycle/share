#include <cmath>
#include <iostream>
#include <chrono>

const int N = 1000010;
const size_t size = N * sizeof(float);

void VecAdd(float* A, float* B, float* C)
{
    for(int i = 0; i < N; i ++)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void VecAddCu(float* A, float* B, float* C, int N)
{
    int i = threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    // generate array:
    float* host_A = new float[N];
    float* host_B = new float[N];
    float* host_C = new float[N];
    float* valid_result = new float[N];

    float* cuda_A;
    cudaMalloc(&cuda_A, size);
    float* cuda_B;
    cudaMalloc(&cuda_B, size);
    float* cuda_C;
    cudaMalloc(&cuda_C, size);

    for(int i = 0; i < N; i ++)
    {
        host_A[i] = ((float) rand() / (RAND_MAX));
        host_B[i] = ((float) rand() / (RAND_MAX));
        host_C[i] = ((float) rand() / (RAND_MAX));
    }
    
    auto start = std::chrono::system_clock::now();

    VecAdd(host_A, host_B, host_C);

    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;
    std::cout << "No cuda: " << elapsed.count() << '\n';

    std::cout << "[INFO]: Start accelerate compute with cuda.\n";
    cudaMemcpy(cuda_A, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, host_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_C, host_C, size, cudaMemcpyHostToDevice);
    
    start = std::chrono::system_clock::now();
    // must be memory allocated by cudaAllocate
    int threadPerBlock = 256;
    int blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;

    VecAddCu<<<blockPerGrid, threadPerBlock>>>(cuda_A, cuda_B, cuda_C, N);

    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "With cuda: " << elapsed.count() << '\n';

    cudaMemcpy(valid_result, cuda_C, size, cudaMemcpyDeviceToHost);
    
    std::cout << "[INFO] Start validate with origin.\n";
    size_t error_num = 0;
    for(int i = 0; i < N; i ++)
    {
        if(std::abs(host_C[i] - valid_result[i]) > 1e-5)
        {
            std::cout << "Error in #" << i << "\n"; 
            error_num ++;
        }
    }

    std::cout << "Error result: " << error_num << std::endl;

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);

    free(host_A);
    free(host_B);
    free(host_C);
    free(valid_result);

    return 0;
}