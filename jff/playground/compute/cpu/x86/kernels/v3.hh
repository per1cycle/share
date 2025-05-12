#include <cstdlib>
namespace solution
{
template<typename T>
void kernel_v3(uint M, uint N, uint K, T *a, T *b, T *c, float alpha, float beta)
{
    constexpr uint BLK = 2;

    for(int i = 0; i < M; i += BLK) 
    {
        for(int j = 0; j < N; j += BLK)
        {
            // cache the result
            float tmp1 = 0.0f;
            float tmp2 = 0.0f;
            float tmp3 = 0.0f;
            float tmp4 = 0.0f;

            // cache the c
            float c1 = c[i * N + j];
            float c2 = c[i * N + j + 1];
            float c3 = c[(i + 1) * N + j];
            float c4 = c[(i + 1) * N + j + 1];
            
            for(int k = 0; k < K; k ++)
            {
                float a1 = 0.0f, a2 = 0.0f;
                float b1 = 0.0f, b2 = 0.0f;
                a1 = a[i * K + k];
                a2 = a[(i + 1) * K + k];
                b1 = b[k * K + j];
                b2 = b[k * K + j + 1];

                tmp1 += a1 * b1;
                tmp2 += a1 * b2;
                tmp3 += a2 * b1;
                tmp4 += a2 * b2;
            }

            c[i * N + j]            = alpha * tmp1 + beta * c1;
            c[i * N + j + 1]        = alpha * tmp2 + beta * c2;
            c[(i + 1) * N + j]      = alpha * tmp3 + beta * c3;
            c[(i + 1) * N + j + 1]  = alpha * tmp4 + beta * c4;
        }
    }
}
}