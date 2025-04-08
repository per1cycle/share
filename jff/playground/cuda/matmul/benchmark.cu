#include <iostream>
#include <vector>

#include "common.cuh"

typedef struct dim
{
    uint n;
    uint m;
    uint k;
} dim_t;

const std::vector<dim_t> dims = 
{
    {.n = 4, .m = 4, .k = 4},
    {.n = 16, .m = 32, .k = 64},
    {.n = 32, .m = 64, .k = 128},
    {.n = 128, .m = 256, .k = 512},
    {.n = 512, .m = 512, .k = 512},
    {.n = 1024, .m = 1024, .k = 1024},
    {.n = 2048, .m = 1024, .k = 1024},
    {.n = 2048, .m = 2048, .k = 2048},
    {.n = 4096, .m = 2048, .k = 2048},
    {.n = 4096, .m = 4096, .k = 4096},
    {.n = 4096, .m = 8192, .k = 8192},
};

const int REP = 50;

void benchmark_info(uint n, uint m, uint k)
{
    std::cout << "Run benchmark with params: " << "N: " \
        << n << ", M: " << m << ", K: " << k << std::endl;
    std::cout << "work load: " << 2.0f * n * m * k * 1e-9 << "GFLOP" << std::endl;
}

void benchmark_result(int N, int M, int K, float total_elapse_time)
{

}

void benchmark()
{
    for(auto dim: dims)
    {
        uint N = dim.n;
        uint M = dim.m;
        uint K = dim.k;

        float *h_a = new float[N * K];
        float *h_b = new float[K * M];
        float *h_c = new float[N * M];

        // benchmark_info(N, M, K);
        gen_data(h_a, h_b, h_c, N, M, K);
        float total_elapse_time = 0.0f;

        for(int r = 0; r < REP; r ++)
        {
            simple_matmul(h_a, h_b, h_c, N, M, K);
        }

        benchmark_result(N, M, K, total_elapse_time);
    }
}

int main()
{
    benchmark();
}