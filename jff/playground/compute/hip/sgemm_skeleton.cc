#include <random>
#include <iostream>

#include <hip/hip_runtime.h>
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


int main(int argc, char ** argv)
{
    if(argc < 2)
    {
        usage();
        exit(1);
    }

    int n = atoi(argv[1]);
    int M = n, N = n, K = n;
    float *h_a = new float[M * K];
    float *h_b = new float[K * N];
    float *h_c = new float[M * N];
    
    init::generate_float_matrix(h_a, M, K);
    init::generate_float_matrix(h_b, M, K);
    init::generate_float_matrix(h_c, M, K);


    delete []h_a;
    delete []h_b;
    delete []h_c;

    return 0;
}