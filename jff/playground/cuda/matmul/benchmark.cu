#include <iostream>
#include <vector>
#include <cstring>
#include <chrono> // for calculate duration.

#include "common.cuh"
#include "kernel/cublas_sgemm.cuh"

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
};

const int REP = 50;

const std::vector<std::string> opt_to_kernel = 
{
   "simple_matmul",
    "cublas_sgemm"
};

void usage()
{
    std::cout << "Run kernel with: " << std::endl;
    std::cout << "\t-0 simple_matmul" << std::endl;
    std::cout << "\t-1 cublas_sgemm" << std::endl;
}

void benchmark_info(uint n, uint m, uint k)
{
    std::cout << "Run benchmark with params: " << "N: " \
        << n << ", M: " << m << ", K: " << k << std::endl;
    std::cout << "work load: " << 2.0f * n * m * k * 1e-9 << " GFLOP" << std::endl;
}

void benchmark_result(int N, int M, int K, float total_elapse_time_in_milisecond)
{
    float total_elapse_time_in_second = total_elapse_time_in_milisecond / REP / 1000.0f; // to seconds.
    std::cout << "FLOPS: " << 2.0f * M * N * K / total_elapse_time_in_second / 1000 / 1000 / 1000 / 1000 << " TFLOPS." << std::endl;
}

void benchmark(uint &opt)
{
    std::cout << "Running with option: " << opt << std::endl;

    for(int i = 0; i < dims.size(); i ++)
    {
        dim_t dim = dims[i];
        uint N = dim.n;
        uint M = dim.m;
        uint K = dim.k;

        float *h_a = new float[N * K];
        float *h_b = new float[K * M];
        float *h_c = new float[N * M];

        // benchmark_info(N, M, K);
        gen_data(h_a, h_b, h_c, N, M, K);
        float total_elapse_time_in_milisecond = 0.0f;

        for(int r = 0; r < REP; r ++)
        {
            float elapse = run_kernel_cublas_sgemm(h_a, h_b, h_c, N, M, K);
            total_elapse_time_in_milisecond += elapse;
        }

        benchmark_result(N, M, K, total_elapse_time_in_milisecond);
    }
}

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        usage();
        exit(1);
    }
    uint opt = atoi(argv[1]);

    benchmark(opt);
}