#ifndef COMMON_CUH
#define COMMON_CUH

#include <random>

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

#endif