#ifndef COMMON_H
#define COMMON_H
#include <random>
#include <iostream>
#include <iomanip>

void gen_data(float *A, float *B, float *C, int N, int M, int K)
{
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < K; j ++)
        {
            A[i * K + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    for(int i = 0; i < K; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            B[i * M + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            C[i * M + j] = 0.0f;
        }
    }
}

void print_arr(float *arr, int row, int col)
{
    for(int r = 0; r < row; r ++)
    {
        for(int c = 0; c < col; c ++)
        {
            std::cout << std::setw(5) << arr[r * col + c] << ' ';
        }
        std::cout << std::endl;
    }
}


void simple_matmul(float *a, float *b, float *c, int N, int M, int K)
{
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            float temp = 0.0f;
            for(int k = 0; k < K; k ++)
            {
                temp += a[i * K + k] * b[k * M + j];
            }
            c[i * M + j] = temp;
        }
    }
}

int compare_result(float *correct, float *compare, int N, int M)
{
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            if(std::abs(correct[i * M + j] - compare[i * M + j]) > 1e-3)
            {
                std::cout << "Error!" << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

#define cuCheck(err) do \
{ \
    if(err != cudaSuccess) { \
        std::cout << "Cuda error: " << cudaGetErrorString(err) << std::endl; \
    } \
} while(0)


#endif // COMMON_H