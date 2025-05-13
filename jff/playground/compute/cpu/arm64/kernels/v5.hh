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

}