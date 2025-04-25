#ifndef COMMON_CUH
#define COMMON_CUH

#include <iostream>
#include <random>
#define abs(x) ((x > 0)? x: - x)

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

#endif