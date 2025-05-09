#include <cstdlib>
namespace solution
{
template<typename T>
void kernel_v1(uint M, uint N, uint K, T *a, T *b, T *c, float alpha, float beta)
{
    for(int i = 0; i < M; i ++) 
    {
        for(int j = 0; j < N; j ++)
        {
            for(int k = 0; k < K; k ++)
            {
                c[i * N + j] = alpha * a[i * K + k] * b[k * N + j] + beta * c[i * N + j];
            }
        }
    }
}
}