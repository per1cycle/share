#include <iostream>

const int N = 1024;
__global__ void MatAdd(float A[N], float B[N], float C[N])
{
    int i = threadIdx.x;
    C[i] = A[i] + B[N - i - 1];
}

int main()
{
    std::cout << "Cuda works" << std::endl;
    float A[N], B[N], C[N];
    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = i;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * sizeof(float), cudaMemcpyHostToDevice);
    MatAdd<<<4, N / 4>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Cuda finished" << std::endl;
}