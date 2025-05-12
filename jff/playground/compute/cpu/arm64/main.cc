#include "common.hh"
#include "kernels/v1.hh"
#include "kernels/v2.hh"
#include "kernels/v3.hh"
#include "kernels/v4.hh"

int main()
{
    constexpr uint M = 2048, N = 2048, K = 2048;
    constexpr uint loop_time = 10;
    float alpha = 1.0f, beta = 0.0f;
    Timer t;

    float *a = (float*) malloc(sizeof(float) * M * K);
    float *b = (float*) malloc(sizeof(float) * K * N);
    float *c = (float*) malloc(sizeof(float) * M * N);
    float *tmp_c = (float*) malloc(sizeof(float) * M * N);
    
    utils::generate_T_matrix<float>(a, M, K);
    utils::generate_T_matrix<float>(b, K, N);
    // kernel v1
    t.start();
    for(int i = 0; i < loop_time; i ++)
    {
        solution::kernel_v1<float>(M, N, K, a, b, c, alpha, beta);
        utils::sgemm_validate_result(c, a, b, tmp_c, M, N, K, alpha, beta);
    }
    t.stop();
    t.report_sgemm_with_loop(M, N, K, alpha, beta, loop_time);
         
    return 0;
}