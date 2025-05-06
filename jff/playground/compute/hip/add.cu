#include <random>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdlib>


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

void cmp_result(float *res, float *a, float *b, int N, int M, int K)
{
    float *tmp = new float[N * K];
    memset(tmp, 0, sizeof(float) * N * K);
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < K; j ++)
        {
            for(int k = 0; k < M; k ++)
            {
                tmp[i * K + j] += a[i * M + k] * b[j + k * K];
            }

            if(abs(res[i * K + j] - tmp[i * K + j]) > 1e-2)
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
__global__ void kernel_add(int M, int N, int K, float *a, float *b, float *c)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    // in the add kernel, M = N = K.
    if(i < M && j < N)
    {
        c[i * N + j] = a[i * K + j] + b[i * N + j];
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

    kernel_add<<<grid_dim, block_dim>>>(M, N, K, d_a, d_b, d_c);

    hipMemcpy(h_c, d_c, M * N * sizeof(float), hipMemcpyDeviceToHost);
    utils::print_array(h_c, M, N);
    
    delete []h_a;
    delete []h_b;
    delete []h_c;

    return 0;
}