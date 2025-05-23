#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.cuh"
template<const uint BM, const uint BN, const uint BK>
__global__ void kernel(uint M, uint N, uint K, half *a, half *b, half *c)
{
    {
        printf("Thread: %d\n", threadIdx.x);
    }
}
int main()
{
    std::cout << "Cuda works" << std::endl;
    int M = 1024, N = 16, K = 1024;

    half *h_a = (half*) malloc(sizeof(half) * M * K);
    half *h_b = (half*) malloc(sizeof(half) * K * N);
    half *h_c = (half*) malloc(sizeof(half) * M * N);

    utils::generate_T_matrix(h_a, M, K);
    utils::generate_T_matrix(h_b, K, N);

    half *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, M * K * sizeof(half));
    cudaMalloc((void **)&d_b, K * N * sizeof(half));
    cudaMalloc((void **)&d_c, M * N * sizeof(half));

    cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_a, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_a, M * N * sizeof(half), cudaMemcpyHostToDevice);
    ///////////////////////////////////////////////////////////////////////////////////////
    // run kernel here
    const uint BM = 16, BN = 16, BK = 512;
    dim3 grid_dim = {M / BM, N / BN, 1};
    dim3 blk_dim = {BM * BN};
    kernel<BM, BN, BK><<<grid_dim, blk_dim>>>(M, N, K, d_a, d_b, d_c);
    ///////////////////////////////////////////////////////////////////////////////////////
    cudaMemcpy(h_c, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
    std::cout << "Cuda finished" << std::endl;
}