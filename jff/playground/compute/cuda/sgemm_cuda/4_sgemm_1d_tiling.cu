#include <iostream>
#include <chrono>

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

/**
 * assuming all of data can be divided by block
 * not fully understand why do this...
 */
template <
    const int BLK_N,
    const int BLK_M,
    const int BLK_K
    >
__global__ void sgemm_1d_tiling(int N, int M, int K, float *a, float *b, float *c, float alpha, float beta)
{
    int c_row = blockIdx.x;
    int c_col = blockIdx.y;

    a += c_row * BLK_M * M;
    b += c_col * BLK_N;
    c += c_row * K * BLK_K + c_col * BLK_N;

    int thread_row = threadIdx.x / BLK;
    int thread_col = threadIdx.x % BLK;
    float tmp = 0.0f;

    __shared__ float a_share[BLK * BLK];
    __shared__ float b_share[BLK * BLK];

    for(int k = 0; k < M; k += BLK)
    {
        a_share[thread_row * BLK + thread_col] = a[thread_row * M + thread_col];
        b_share[thread_row * BLK + thread_col] = b[thread_row * K + thread_col];

        __syncthreads();

        a += BLK;
        b += BLK * K;

        for(int inner = 0; inner < BLK; inner ++)
        {
            tmp += a_share[thread_row * BLK + inner] * b_share[inner * BLK + thread_col];
        }

        __syncthreads();

    }

    c[thread_row * K + thread_col] = alpha * tmp + beta;
}
/**
 * Time:           0.63702ms.
 * GFlop:          137.456
 * GFLOPS:         215.779
 * Percentage:     4.9321%.
 */
int main(int argc, char ** argv)
{
    if(argc < 2)
    {
        usage();
        exit(1);
    }
    uint N;
    const int BLK_N = 32;
    const int BLK_M = 32;
    const int BLK_K = 8;

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
    dim3 grid_dim = {N / BLK_N, N / BLK_M};
    dim3 blk_dim = {BLK_N * BLK_M};

    // start measuring.
    float elapsed; // in milisecond

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sgemm_1d_tiling<BLK_N, BLK_M, BLK_K><<<grid_dim, blk_dim>>>(N, N, N, d_a, d_b, d_c, 1.0f, 0.0f);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    // std::cout << "GFLOPS: " << flops / 1000000000.0f / elapsed * 1000.0f << std::endl;

    float gflop = flop / 1000000000.0f;
    elapsed = elapsed / 1000.0f; // to second

    std::cout 
            << "Time:                                   \t"   << elapsed << "ms.\n"
            << "GFlop:                                  \t"   << gflop << "\n"
            << "GFLOPS:                                 \t"     << gflop / elapsed << "\n"
            << "Percentage(compare to theoratical peak):\t"     << (gflop / elapsed) / 4591.26f * 100.0 << "%.\n"
            << "Percentage(compare to cublas peak):     \t"   << (gflop / elapsed) / 3815.45f * 100.0 << "%.\n";

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cmp_result(h_c, h_a, h_b, N, N, N);
    return 0;
}