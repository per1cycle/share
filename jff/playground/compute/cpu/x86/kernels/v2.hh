#include <cstdlib>
namespace solution
{
template<typename T>
void kernel_v2(uint M, uint N, uint K, T *a, T *b, T *c, float alpha, float beta)
{
    for(int i = 0; i < M; i ++) 
    {
        for(int j = 0; j < N; j ++)
        {
            float tmp = 0.0f;
            for(int k = 0; k < K; k ++)
            {
                tmp += a[i * K + k] * b[k * K + j];
            }
            c[i * N + j] = alpha * tmp + beta * c[i * N + j];
        }
    }
}
}