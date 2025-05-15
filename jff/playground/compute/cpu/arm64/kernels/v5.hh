#include <iostream>
#include <arm_neon.h>
#include <iostream>

namespace solution
{
void play()
{
    float32_t a[4] = {1.0, 2.0, 3.0, 4.0};
    float32_t b[4] = {6.0, 2.0, 3.2, 4.0};
    float32_t res[4];

    float32x4_t reg_a = vld1q_f32(a);
    float32x4_t reg_b = vld1q_f32(b);

    float32x4_t reg_res = vaddq_f32(reg_a, reg_b);

    vst1q_f32(res, reg_res);
}
template<
    typename T>
void kernel_v5(uint M, uint N, uint K, T *a, T *b, T *c, float alpha, float beta)
{
    
    for(int i = 0; i < M; i += 4) 
    {
        for(int j = 0; j < N; j += 4)
        {
            float32x4_t tmp1, tmp2, tmp3, tmp4;

            for(int k = 0; k < K; k ++)
            {
                
                float32x4_t b1 = vld1q_f32(&b[k * N + j]);
                float a1 = a[(i) * K + k];
                float a2 = a[(i + 1) * K + k];
                float a3 = a[(i + 2) * K + k];
                float a4 = a[(i + 3) * K + k];

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