#include <hip/hip_runtime.h>
#include <kernel/kernels.cuh>
#include <cstdlib>
#include <iostream>
#include <string>

int main()
{
    int M = 4096, N = 4096, K = 4096;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float*)malloc(M * K * sizeof(float));
    h_b = (float*)malloc(K * N * sizeof(float));
    h_c = (float*)malloc(M * N * sizeof(float));
    init::generate_float_matrix(h_a, M, K);
    init::generate_float_matrix(h_b, K, N);

    hipMalloc((void**)&d_a, M * K * sizeof(float));
    hipMalloc((void**)&d_b, K * N * sizeof(float));
    hipMalloc((void**)&d_c, M * N * sizeof(float));

    return 0;
}