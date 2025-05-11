#include <cstdlib>
namespace solution
{
template<
    typename T>
void kernel_v3(uint M, uint N, uint K, T *a, T *b, T *c, float alpha, float beta)
{
    
    for(int i = 0; i < M; i += 2) 
    {
        for(int j = 0; j < N; j += 2)
        {
            float tmp1 = 0.0f;   
            float tmp2 = 0.0f;   
            float tmp3 = 0.0f;   
            float tmp4 = 0.0f;

            for(int k = 0; k < K; k ++)
            {
                float a1 = a[(i) * K + k];
                float a2 = a[(i + 1) * K + k];

                float b1 = b[(k) * N + j];
                float b2 = b[(k) * N + j + 1];

                tmp1 += a1 * b1;
                tmp2 += a1 * b2;
                tmp3 += a2 * b1;
                tmp4 += a2 * b2;
            }

            c[(i) * N + (j)] = alpha * tmp1 + beta * c[(i) * N + (j)];
            c[(i) * N + (j + 1)] = alpha * tmp2 + beta * c[(i) * N + (j + 1)];
            c[(i + 1) * N + (j)] = alpha * tmp3 + beta * c[(i + 1) * N + (j)];
            c[(i + 1) * N + (j + 1)] = alpha * tmp4 + beta * c[(i + 1) * N + (j + 1)];
        }
    }
}

}