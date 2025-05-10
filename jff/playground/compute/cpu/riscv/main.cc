#include "common.hh"
#include "kernels/v1.hh"

int main()
{
    constexpr uint M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 1.0f;
    Timer t;

    float *a = (float*) malloc(sizeof(float) * M * K);
    float *b = (float*) malloc(sizeof(float) * K * N);
    float *c = (float*) malloc(sizeof(float) * M * N);
    
    utils::generate_T_matrix<float>(a, M, K);
    utils::generate_T_matrix<float>(b, K, N);
    t.start();
    solution::kernel_v1<float>(M, N, K, a, b, c, alpha, beta);
    t.stop();
    return 0;
}