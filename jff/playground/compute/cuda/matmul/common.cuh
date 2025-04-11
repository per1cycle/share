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

void gen_data_in_row_major(float *A, float *B, float *C, int N, int M, int K)
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

void gen_data_in_col_major(float *A, float *B, float *C, int N, int M, int K)
{
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < K; j ++)
        {
            A[j * N + i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    for(int i = 0; i < K; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            B[j * K + i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }
    
    // only need to set all to zero.
    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            C[i * M + j] = 0.0f;
        }
    }
}

void print_arr(float *a, int row, int col, std::string arr_name)
{
    std::cout << arr_name << " = np.array(";
    std::cout << "[";
    for(int i = 0; i < row; i ++)
    {
        std::cout << '[';
        for(int j = 0; j < col; j ++)
        {
            std::cout << a[i * col + j] << ", ";
        }
        if(i != row - 1)
            std::cout << "]," << std::endl;
        else 
            std::cout << "]";
    }
    std::cout << "])" << std::endl;
}

void simple_matmul_row_major(float *a, float *b, float *c, int N, int M, int K);

void simple_matmul(float *a, float *b, float *c, int N, int M, int K)
{
    simple_matmul_row_major(a, b, c, N, M, K);
}

void simple_matmul_row_major(float *a, float *b, float *c, int N, int M, int K)
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
/**
 * mul two matrix in column major way
 * e.g: a [
 *      N is the row number used in row major arrangement
 *      K is the column number used in row major arrangement
 * ]
 */
void simple_matmul_col_major(float *a, float *b, float *c, int N, int M, int K)
{
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