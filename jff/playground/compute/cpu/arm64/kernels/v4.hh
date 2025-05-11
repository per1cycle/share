#include <cstdlib>
namespace solution
{
template<
    typename T>
void kernel_v4(uint M, uint N, uint K, T *a, T *b, T *c, float alpha, float beta)
{
    
    for(int i = 0; i < M; i += 4) 
    {
        for(int j = 0; j < N; j += 4)
        {
            float tmp11 = 0.0f;   
            float tmp12 = 0.0f;   
            float tmp13 = 0.0f;   
            float tmp14 = 0.0f;

            float tmp21 = 0.0f;   
            float tmp22 = 0.0f;   
            float tmp23 = 0.0f;   
            float tmp24 = 0.0f;

            float tmp31 = 0.0f;   
            float tmp32 = 0.0f;   
            float tmp33 = 0.0f;   
            float tmp34 = 0.0f;

            float tmp41 = 0.0f;   
            float tmp42 = 0.0f;   
            float tmp43 = 0.0f;   
            float tmp44 = 0.0f;

            for(int k = 0; k < K; k ++)
            {
                float a1 = a[(i) * K + k];
                float a2 = a[(i + 1) * K + k];
                float a3 = a[(i + 2) * K + k];
                float a4 = a[(i + 3) * K + k];

                float b1 = b[(k) * N + j];
                float b2 = b[(k) * N + j + 1];
                float b3 = b[(k) * N + j + 2];
                float b4 = b[(k) * N + j + 3];

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

            c[(i) * N + (j)] = alpha * tmp11 + beta * c[(i) * N + (j)];
            c[(i) * N + (j + 1)] = alpha * tmp12 + beta * c[(i) * N + (j + 1)];
            c[(i) * N + (j + 2)] = alpha * tmp13 + beta * c[(i) * N + (j + 2)];
            c[(i) * N + (j + 3)] = alpha * tmp14 + beta * c[(i) * N + (j + 3)];

            c[(i + 1) * N + (j)] = alpha * tmp21 + beta * c[(i + 1) * N + (j)];
            c[(i + 1) * N + (j + 1)] = alpha * tmp22 + beta * c[(i + 1) * N + (j + 1)];
            c[(i + 1) * N + (j + 2)] = alpha * tmp23 + beta * c[(i + 1) * N + (j + 2)];
            c[(i + 1) * N + (j + 3)] = alpha * tmp24 + beta * c[(i + 1) * N + (j + 3)];

            c[(i + 2) * N + (j)] = alpha * tmp31 + beta * c[(i + 2) * N + (j)];
            c[(i + 2) * N + (j + 1)] = alpha * tmp32 + beta * c[(i + 2) * N + (j + 1)];
            c[(i + 2) * N + (j + 2)] = alpha * tmp33 + beta * c[(i + 2) * N + (j + 2)];
            c[(i + 2) * N + (j + 3)] = alpha * tmp34 + beta * c[(i + 2) * N + (j + 3)];

            c[(i + 3) * N + (j)] = alpha * tmp41 + beta * c[(i + 3) * N + (j)];
            c[(i + 3) * N + (j + 1)] = alpha * tmp42 + beta * c[(i + 3) * N + (j + 1)];
            c[(i + 3) * N + (j + 2)] = alpha * tmp43 + beta * c[(i + 3) * N + (j + 2)];
            c[(i + 3) * N + (j + 3)] = alpha * tmp44 + beta * c[(i + 3) * N + (j + 3)];

        }
    }
}

}