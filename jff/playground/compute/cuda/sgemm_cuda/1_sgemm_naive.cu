#include <iostream>
#include <chrono>

#include "common.cuh"

int N;
uint BLK = 32;

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

/**
 * assuming all of data can be divided by block
 */
__global__ void sgemm_naive(int N, int M, int K, float *a, float *b, float *c, float alpha, float beta)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x <= N && y <= K)
    {
        float tmp = 0.0f;
        for(int k = 0; k < M; k ++)
        {
            tmp += a[x * M + k] * b[k * K + y];
        }

        c[x * K + y] = alpha * tmp + beta;
    }
}
/**
 * Time:           0.524241ms.
 *  GFlop:          17.1841
 *  GFLOPS:         32.7789
 *  Percentage:     0.749233%.
 */
int main(int argc, char ** argv)
{
    if(argc < 2)
    {
        usage();
        exit(1);
    }

    N = atoi(argv[1]);
    size_t size = sizeof(float) * N * N;
    float flop = 1.0 * N * N * (2 * N + 1);
    
    float *h_a = new float[N * N];
    float *h_b = new float[N * N];
    float *h_c = new float[N * N];

    generate_float_matrix(h_a, N, N);
    generate_float_matrix(h_b, N, N);
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

    // fix typo.
    dim3 grid_dim = {N / BLK, N / BLK, 1};
    dim3 blk_dim = {BLK, BLK, 1};

    // start measuring.
    float elapsed; // in milisecond

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sgemm_naive<<<grid_dim, blk_dim>>>(N, N, N, d_a, d_b, d_c, 1.0f, 0.0f);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    // std::cout << "GFLOPS: " << flops / 1000000000.0f / elapsed * 1000.0f << std::endl;
    
    float gflop = flop / 1000000000.0f;
    elapsed = elapsed / 1000.0f; // to second

    std::cout 
            << "Time:                                   \t" << elapsed << "ms.\n"
            << "GFlop:                                  \t" << gflop << "\n"
            << "GFLOPS:                                 \t" << gflop / elapsed << "\n"
            << "Percentage(compare to theoratical peak):\t" << (gflop / elapsed) / 4591.26f * 100.0 << "%.\n"
            << "Percentage(compare to cublas peak):     \t" << (gflop / elapsed) / 3737.3f * 100.0 << "%.\n";

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    return 0;
}