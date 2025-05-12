#include <cstdlib>
namespace solution
{
template<typename T>
void kernel_v4(uint M, uint N, uint K, T *a, T *b, T *c, float alpha, float beta)
{
    constexpr uint BLK = 4;

    for(int i = 0; i < M; i += BLK) 
    {
        for(int j = 0; j < N; j += BLK)
        {
            T tmp11 = 0.0f;
            T tmp12 = 0.0f;
            T tmp13 = 0.0f;
            T tmp14 = 0.0f;
            
            T tmp21 = 0.0f;
            T tmp22 = 0.0f;
            T tmp23 = 0.0f;
            T tmp24 = 0.0f;
            
            T tmp31 = 0.0f;
            T tmp32 = 0.0f;
            T tmp33 = 0.0f;
            T tmp34 = 0.0f;

            T tmp41 = 0.0f;
            T tmp42 = 0.0f;
            T tmp43 = 0.0f;
            T tmp44 = 0.0f;
            ///////////////////

            T c11 = c[i * N + j];
            T c12 = c[i * N + j + 1];
            T c13 = c[i * N + j + 2];
            T c14 = c[i * N + j + 3];
            
            T c21 = c[(i + 1) * N + j];
            T c22 = c[(i + 1) * N + j + 1];
            T c23 = c[(i + 1) * N + j + 2];
            T c24 = c[(i + 1) * N + j + 3];
            
            T c31 = c[(i + 2) * N + j];
            T c32 = c[(i + 2) * N + j + 1];
            T c33 = c[(i + 2) * N + j + 2];
            T c34 = c[(i + 2) * N + j + 3];

            T c41 = c[(i + 3) * N + j];
            T c42 = c[(i + 3) * N + j + 1];
            T c43 = c[(i + 3) * N + j + 2];
            T c44 = c[(i + 3) * N + j + 3];
            
            for(int k = 0; k < K; k ++)
            {
                T a1 = a[i * K + k];
                T a2 = a[(i + 1) * K + k];
                T a3 = a[(i + 2) * K + k];
                T a4 = a[(i + 3) * K + k];

                T b1 = b[k * K + j];
                T b2 = b[k * K + j + 1];
                T b3 = b[k * K + j + 2];
                T b4 = b[k * K + j + 3];

                tmp11 += a1 * b1;
                tmp12 += a1 * b2;
                tmp13 += a1 * b3;
                tmp14 += a1 * b4;

                tmp21 += a2 * b1;
                tmp22 += a2 * b2;
                tmp23 += a2 * b3;
                tmp24 += a2 * b4;

                tmp31 += a3 * b1;
                tmp32 += a3 * b2;
                tmp33 += a3 * b3;
                tmp34 += a3 * b4;

                tmp41 += a4 * b1;
                tmp42 += a4 * b2;
                tmp43 += a4 * b3;
                tmp44 += a4 * b4;
            }

            c[i * N + j]        = alpha * tmp11 + beta * c11;
            c[i * N + j + 1]    = alpha * tmp12 + beta * c12;
            c[i * N + j + 2]    = alpha * tmp13 + beta * c13;
            c[i * N + j + 3]    = alpha * tmp14 + beta * c14;

            c[(i + 1) * N + j]        = alpha * tmp21 + beta * c21;
            c[(i + 1) * N + j + 1]    = alpha * tmp22 + beta * c22;
            c[(i + 1) * N + j + 2]    = alpha * tmp23 + beta * c23;
            c[(i + 1) * N + j + 3]    = alpha * tmp24 + beta * c24;

            c[(i + 2) * N + j]        = alpha * tmp31 + beta * c31;
            c[(i + 2) * N + j + 1]    = alpha * tmp32 + beta * c32;
            c[(i + 2) * N + j + 2]    = alpha * tmp33 + beta * c33;
            c[(i + 2) * N + j + 3]    = alpha * tmp34 + beta * c34;

            c[(i + 3) * N + j]        = alpha * tmp41 + beta * c41;
            c[(i + 3) * N + j + 1]    = alpha * tmp42 + beta * c42;
            c[(i + 3) * N + j + 2]    = alpha * tmp43 + beta * c43;
            c[(i + 3) * N + j + 3]    = alpha * tmp44 + beta * c44;
        }
    }
}
}