#include <iostream>
// #include <common.cuh>
int main()
{
    std::cout << "Cuda works" << std::endl;
    int M = 1024, N = 1024, K = 1024;

    float *h_a = (float*) malloc(sizeof(float) * M * K);
    float *h_b = (float*) malloc(sizeof(float) * K * N);
    float *h_c = (float*) malloc(sizeof(float) * M * N);

    float *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, M * K * sizeof(float));
    cudaMalloc((void **)&d_b, K * N * sizeof(float));
    cudaMalloc((void **)&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    ///////////////////////////////////////////////////////////////////////////////////////
    // run kernel here

    ///////////////////////////////////////////////////////////////////////////////////////
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
    std::cout << "Cuda finished" << std::endl;
}