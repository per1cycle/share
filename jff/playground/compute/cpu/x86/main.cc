#include "common.hh"
#include "kernels/v1.hh"
#include "kernels/v2.hh"
#include "kernels/v3.hh"
#include "kernels/v4.hh"
#include "kernels/v5.hh"

int main()
{
    constexpr uint M = 1024, N = 1024, K = 1024;
    int loop = 10, kernel_num = 2;
    float alpha = 1.0f, beta = 0.0f;
    Timer t;

    float *a = (float*) malloc(sizeof(float) * M * K);
    float *b = (float*) malloc(sizeof(float) * K * N);
    float *c = (float*) malloc(sizeof(float) * M * N);

    utils::generate_T_matrix<float>(a, M, K);
    utils::generate_T_matrix<float>(b, K, N);

    for(int i = 0; i < loop; i ++)
    {
        t.start();
        // solution::kernel_v4<float>(M, N, K, a, b, c, alpha, beta);
        solution::play();
        t.stop();
        // utils::cmp_result<float>(c, a, b, M, N, K);
        // utils::print_array(c, M, N);
        // t.report_sgemm(M, N, K, alpha, beta);
        t.just_report_time();
        t.reset();
    }

    return 0;
}