#include <iostream>
#include <chrono>

#include "common.cuh"
#include <assert.h>

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
 * completely doesnot understand this.
 */
template <
    const int BLK_N,
    const int BLK_M, // reduce the m to optimize blkm
    const int BLK_K,
    const int THREAD_N,
    const int THREAD_K
    >
__global__ void sgemm_2d_tiling(int N, int M, int K, float *a, float *b, float *c, float alpha, float beta)
{
    int c_row = blockIdx.x;
    int c_col = blockIdx.y;

    // the total tile of the result block.
    int total_result = BLK_N * BLK_K;
    // total tile number
    int num_thread_blocktile = total_result / (THREAD_N * THREAD_K);

    assert(num_thread_blocktile == blockDim.x);

    int thread_row = threadIdx.x / (BLK_K / THREAD_K);
    int thread_col = threadIdx.x % (BLK_K / THREAD_K);

    int stride_a = num_thread_blocktile / BLK_M;
    int inner_a_row = threadIdx.x / BLK_M;
    int inner_a_col = threadIdx.x % BLK_M;

    int stride_b = num_thread_blocktile / BLK_K;
    int inner_b_row = threadIdx.x / BLK_K;
    int inner_b_col = threadIdx.x % BLK_K;

    a += c_row * BLK_N * M;
    b += c_col * BLK_K;
    c += c_row * K * BLK_N + c_col * BLK_K;

    __shared__ float a_share[BLK_N * BLK_M];
    __shared__ float b_share[BLK_M * BLK_K];

    float thread_res[THREAD_N * THREAD_K] = {0.0f};

    float register_n[THREAD_N] = {0.0f};
    float register_k[THREAD_K] = {0.0f};

    /**
     * A: N * M
     * B: M * K
     */
    for(int blk_m = 0; blk_m < M; blk_m += BLK_M)
    {
        // move to share memory cache.
        for(int offset = 0; offset < BLK_N; offset += stride_a)
            a_share[(inner_a_row + offset) * BLK_M + inner_a_col] 
                = a[(inner_a_row + offset) * M + inner_a_col];

        for(int offset = 0; offset < BLK_M; offset += stride_b)
            b_share[(inner_b_row + offset) * BLK_K + inner_b_col] 
                = b[(inner_b_row + offset) * K + inner_b_col];
        
        __syncthreads();

        a += BLK_M;
        b += BLK_M * K;
        
        // calculate teh thread result
        for(int inner_m = 0; inner_m < BLK_M; inner_m ++)
        {
            // move share memory cache to register.
            for(int register_a_idx = 0; register_a_idx < THREAD_N; register_a_idx ++)
                register_n[register_a_idx] = a_share[(thread_row * THREAD_K + register_a_idx) * BLK_M + inner_m];
            
            for(int register_b_idx = 0; register_b_idx < THREAD_K; register_b_idx ++)
                register_k[register_b_idx] = b_share[inner_m * BLK_K + thread_col * THREAD_K + register_b_idx];

            for(int tid_n = 0; tid_n < THREAD_N; tid_n++)
            {
                for(int tid_k = 0; tid_k < THREAD_K; tid_k ++)
                {
                    thread_res[tid_n * THREAD_K + tid_k] += register_n[tid_n] * register_k[tid_k];
                }
            }
        }
        __syncthreads();
    }

    // write result back to c array.
    for(int tid_n = 0; tid_n < THREAD_N; tid_n ++)
        for(int tid_k = 0; tid_k < THREAD_K; tid_k ++)
        {
            c[(thread_row * THREAD_N + tid_n) * K + thread_col * THREAD_K + tid_k]
                = alpha * thread_res[tid_n * THREAD_K + tid_k] + beta;
        }
}

/**
 * Time:                                           0.113557ms.
 * GFlop:                                          137.456
 * GFLOPS:                                         1210.46
 * Percentage(compare to theoratical peak):        26.3644%.
 * Percentage(compare to cublas peak):             31.7252%.
 */
int main(int argc, char ** argv)
{
    if(argc < 2)
    {
        usage();
        exit(1);
    }
    uint N;
    const int BLK_N = 64;
    const int BLK_M = 8;
    const int BLK_K = 64;
    const int THREAD_N = 8;
    const int THREAD_K = 8;

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
    dim3 grid_dim = {N / BLK_N, N / BLK_K};
    dim3 blk_dim = {(BLK_N * BLK_K) / (THREAD_N * THREAD_K)};

    // start measuring.
    float elapsed; // in milisecond

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sgemm_2d_tiling<BLK_N, BLK_M, BLK_K, THREAD_N, THREAD_K><<<grid_dim, blk_dim>>>(N, N, N, d_a, d_b, d_c, 1.0f, 0.0f);

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