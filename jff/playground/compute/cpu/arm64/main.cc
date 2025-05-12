#include "common.hh"
#include "kernels/v1.hh"
#include "kernels/v2.hh"
#include "kernels/v3.hh"
#include "kernels/v4.hh"

int main()
{
    constexpr uint M = 1024, N = 1024, K = 1024;
    constexpr uint loop_time = 10;
    float alpha = 1.0f, beta = 0.0f;
    Timer t;

    float *a = (float*) malloc(sizeof(float) * M * K);
    float *b = (float*) malloc(sizeof(float) * K * N);
    float *c = (float*) malloc(sizeof(float) * M * N);
    
    utils::generate_T_matrix<float>(a, M, K);
    utils::generate_T_matrix<float>(b, K, N);
    // kernel v1
    t.start();
    for(int i = 0; i < loop_time; i ++)
        solution::kernel_v1<float>(M, N, K, a, b, c, alpha, beta);
    t.stop();
    // utils::cmp_result(c, a, b, M, N, K);
    t.report_sgemm_with_loop(M, N, K, alpha, beta, loop_time);
    t.reset();

    // kernel v2
    t.start();
    for(int i = 0; i < loop_time; i ++)
        solution::kernel_v2<float>(M, N, K, a, b, c, alpha, beta);
    t.stop();
    // utils::cmp_result(c, a, b, M, N, K);
    t.report_sgemm_with_loop(M, N, K, alpha, beta, loop_time);
    t.reset();

    // kernel v3
    t.start();
    for(int i = 0; i < loop_time; i ++)
        solution::kernel_v3<float>(M, N, K, a, b, c, alpha, beta);
    t.stop();
    // utils::cmp_result(c, a, b, M, N, K);
    t.report_sgemm_with_loop(M, N, K, alpha, beta, loop_time);
    t.reset();
    
    // kernel v4
    t.start();
    for(int i = 0; i < loop_time; i ++)
        solution::kernel_v4<float>(M, N, K, a, b, c, alpha, beta);
    t.stop();
    // utils::cmp_result(c, a, b, M, N, K);
    t.report_sgemm_with_loop(M, N, K, alpha, beta, loop_time);
    t.reset();
    
    return 0;
}