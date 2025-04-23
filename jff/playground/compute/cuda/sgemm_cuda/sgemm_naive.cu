#include <iostream>
#include "common.cuh"

void usage()
{
    std::cout << "Usage: ./a.out <matrix dimension>\n"
                << "Notice, the a x b = c\n"
                << "a: N x M matrix\n"
                << "b: M x K matrix\n"
                << "c: N x K matrix\n"
                // << "to align with cublas, we use b^T x a^T = c^T\n"
                // << "aka: b K x M matrix, a M x N matrix, c K x N matrix\n"
                << "In the example here the N = M = K\n";
}

__global__ void sgemm_naive(int N, int M, int K, float *a, float *b, float *c)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    float tmp = 0.0f;
    for(int k = 0; k < M; k ++)
    {
        tmp += a[x * M + k] * b[y + k * K];
    }

    c[x * K + y] = tmp;
}

int main(int argc, char ** argv)
{
    if(argc < 2)
    {
        usage();
        exit(1);
    }

    int N = atoi(argv[1]);
    uint BLK = 32;
    size_t size = sizeof(float) * N * N;
    
    float *h_a = new float[N * N];
    float *h_b = new float[N * N];
    float *h_c = new float[N * N];

    generate_float_matrix(h_a, N, N);
    generate_float_matrix(h_b, N, N);
    print_array(h_a, N, N);
    memset(h_c, 0, size);

    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);

    dim3 grid_dim = {BLK, BLK, 1};
    dim3 blk_dim = {N / BLK, N / BLK, 1};

    sgemm_naive<<<grid_dim, blk_dim>>>(N, N, N, d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    print_array(h_c, N, N);
    return 0;
}