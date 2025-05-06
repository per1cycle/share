#include <hip/hip_runtime.h>
#include "kernel/kernels.cuh"
#include "utils/utils.cuh"
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        std::cout << "Usage: ./a.out kernel_type" << std::endl;
        return 1;
    }
    int M = 4096, N = 4096, K = 4096;
    float gflop = 1.0f * (M * N * K * 2 + M * N * 3) * 1e-9;

    
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    Timer timer;

    h_a = (float*)malloc(M * K * sizeof(float));
    h_b = (float*)malloc(K * N * sizeof(float));
    h_c = (float*)malloc(M * N * sizeof(float));
    init::generate_float_matrix(h_a, M, K);
    init::generate_float_matrix(h_b, K, N);

    hipMalloc((void**)&d_a, M * K * sizeof(float));
    hipMalloc((void**)&d_b, K * N * sizeof(float));
    hipMalloc((void**)&d_c, M * N * sizeof(float));

    hipMemcpy(d_a, h_a, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, K * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_c, h_c, M * N * sizeof(float), hipMemcpyHostToDevice);

    int kernel_type = atoi(argv[1]);

    timer.start();
    run_kernel(kernel_type, d_a, d_b, d_c, M, N, K);
    timer.stop();

    std::cout 
            << "Time:                                   \t" << timer.elapsed() << "s.\n"
            << "GFlop:                                  \t" << gflop << "\n"
            << "GFLOPS:                                 \t" << gflop / timer.elapsed() << "\n";

    hipMemcpy(h_c, d_c, M * N * sizeof(float), hipMemcpyDeviceToHost);

    return 0;
}