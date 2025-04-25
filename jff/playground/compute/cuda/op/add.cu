#include <iostream>

const int N = 1024 * 1024;
template <const int NUM>
__global__ void MatAdd(float A[N], float B[N], float *C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < NUM)
        atomicAdd(C, A[i] + B[i]);
}

int main()
{
    std::cout << "Cuda works" << std::endl;
    float *A, *B, C[1];
    A = (float*)malloc(sizeof(float) * N);
    B = (float*)malloc(sizeof(float) * N);
    C[0] = 0.0f;
    for (int i = 0; i < N; i++)
    {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, 1 * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(float), cudaMemcpyHostToDevice);

    MatAdd<N><<<N / 1024, 1024>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << C[0] << std::endl;
    std::cout << "Cuda finished" << std::endl;
}