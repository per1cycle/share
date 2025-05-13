#include "common.hh"
#include "kernels/v1.hh"
#include "kernels/v2.hh"
#include "kernels/v3.hh"
#include "kernels/v4.hh"
#include "kernels/v5.hh"

int main()
{
    constexpr uint M = 2048, N = 2048, K = 2048;
    constexpr uint loop_time = 1000000;
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
        solution::play();
    }
    t.stop();
    t.report_sgemm_with_loop(M, N, K, alpha, beta, loop_time);
         
    return 0;
}