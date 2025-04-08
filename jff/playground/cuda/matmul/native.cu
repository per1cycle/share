#include <cuda_runtime.h>
#include "common.cuh"

// const int NUM_GROUP = 6;
// const int REPEAT = 30;

// int groups[6][3] = 
// {
//     {16, 16, 16},
//     {128, 128, 128},
//     {1024, 1024, 1024},
//     {2048, 2048, 2048},
//     {4096, 4096, 4096},
//     {8192, 8192, 8192},
// };

/**
 * @brief Each __global__ function runs on one grid.
 * in this simple situation we don't consider the outbound condition
 */
__global__ void native(float *A, float *B, float *C, int N, int M, int K)
{
    // Calculate global row and column indices correctly
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Make sure we're not out of bounds
    if(row < N && col < M)
    {
        float tmp = 0;
        for(int k = 0; k < K; k++)
            tmp += A[row * K + k] * B[k * M + col];
        C[row * M + col] = tmp;
    }
}
int main()
{
    const int N = 4096, M = 4096, K = 4096;
    const int BS = 32;
    float flops = 2.0f * N * M * K;

    float *h_a = new float[N * K];
    float *h_b = new float[K * M];
    float *h_c = new float[N * M];
    gen_data(h_a, h_b, h_c, N, M, K);
    
    float *d_a;
    float *d_b;
    float *d_c;
    cuCheck(cudaMalloc((void **)&d_a, N * K * sizeof(float)));
    cuCheck(cudaMalloc((void **)&d_b, K * M * sizeof(float)));
    cuCheck(cudaMalloc((void **)&d_c, N * M * sizeof(float)));

    cuCheck(cudaMemcpy(d_a, h_a, N * K * sizeof(float), cudaMemcpyHostToDevice));
    cuCheck(cudaMemcpy(d_b, h_b, K * M * sizeof(float), cudaMemcpyHostToDevice));
    cuCheck(cudaMemcpy(d_c, h_c, N * M * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid_dim(N / BS, M / BS, 1); // there's N / bs blocks in grid x dim, M / BS in grid y dim, 1 in grid z dim.
    dim3 block_dim(BS, BS, 1); // there's 32 threads in block x dim, 32 threads in block y dim, 1 thread in block z dim.

    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    native<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N, M, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    et = et * 1e-3; // convert to seconds.
    std::cout << "FLOPS: " << (flops) * 1e-9 / et << " GFLOPS." << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}