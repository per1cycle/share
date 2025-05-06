#include <random>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdlib>

#define abs(x) (((x) > 0)? (x): - (x))

namespace init
{

void generate_float_matrix(float *out, int row, int column)
{
    srand(time(NULL));
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            out[i * column + j] = (float)rand() / (float)(INT32_MAX);
        }
    }
}

}

namespace utils
{
void print_array(float *arr, int row, int column)
{
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            std::cout << arr[i * column + j] << ' ';
        }
        std::cout << std::endl;
    }
}

void cmp_result(float *res, float *a, float *b, int M, int N, int K)
{

    for(int i = 0; i < M; i ++) // i th row
    {
        for(int j = 0; j < N; j ++) // j th column
        {
            float tmp = 0.0f;
            for(int k = 0; k < K; k ++)
            {
                tmp += a[i * K + k] * b[j + k * N];
            }

            if(abs(res[i * K + j] - tmp) > 1e-3)
            {
                std::cout << "Result error.\n";
                exit(1);
            }
        }
    }
    std::cout << "Correct! be proud of it!\n";

}

}

void usage()
{
    std::cout
        << "Usage: ./a.out <N>"
        << std::endl;
}

/**
 * add two matrix to c, notice in the situation the M = N = K
 */
__global__ void kernel_naive(int M, int N, int K, float *a, float *b, float *c, float alpha, float beta)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        float acc_c = 0.0f; 
        for (int k = 0; k < K; ++k)
        {
            acc_c += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = alpha * acc_c + beta * c[row * N + col];
    }
}

int main(int argc, char ** argv)
{
    if(argc < 2)
    {
        usage();
        exit(1);
    }

    uint n = atoi(argv[1]);
    // add two 256 x 256
    uint M = n, N = n, K = n;
    float *h_a = new float[M * K];
    float *h_b = new float[K * N];
    float *h_c = new float[M * N];
    float elapsed = 0.0f;
    float flop = 1.0 * N * N * (2 * N + 1);

    init::generate_float_matrix(h_a, M, K);
    init::generate_float_matrix(h_b, K, N);
    memset(h_c, 0, sizeof(float) * M * N);

    float *d_a, *d_b, *d_c;

    hipMalloc((void **)&d_a, M * K * sizeof(float));
    hipMalloc((void **)&d_b, K * N * sizeof(float));
    hipMalloc((void **)&d_c, M * N * sizeof(float));

    hipMemcpy(d_a, h_a, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, K * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_c, h_c, M * N * sizeof(float), hipMemcpyHostToDevice);

    dim3 grid_dim = {(N / 32), (N / 32), 1};
    dim3 block_dim = {32, 32, 1};

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    kernel_naive<<<grid_dim, block_dim>>>(M, N, K, d_a, d_b, d_c, 1.0f, 0.0f);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed, start, stop);

    float gflop = flop / 1000000000.0f;
    elapsed = elapsed / 1000.0f; // to second

    std::cout 
            << "Time:                                   \t" << elapsed << "ms.\n"
            << "GFlop:                                  \t" << gflop << "\n"
            << "GFLOPS:                                 \t" << gflop / elapsed << "\n";

    hipMemcpy(h_c, d_c, M * N * sizeof(float), hipMemcpyDeviceToHost);

    utils::cmp_result(h_c, h_a, h_b, M, N, K);
    delete []h_a;
    delete []h_b;
    delete []h_c;

    return 0;
}