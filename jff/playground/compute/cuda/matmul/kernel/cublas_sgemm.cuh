#ifndef CUBLAS_SGEMM_H
#define CUBLAS_SGEMM_H
#include <cublas_v2.h>
#include <chrono>
#include <iostream>

float run_kernel_cublas_sgemm(float *h_a, float *h_b, float *h_c, int N, int M, int K)
{
    float *d_a;
    float *d_b;
    float *d_c;
    float alpha = 1.0f;
    float beta = 0.0f;  // This should be 0.0f for a clean multiplication
    
    (cudaMalloc((void **)&d_a, N * K * sizeof(float)));
    (cudaMalloc((void **)&d_b, K * M * sizeof(float)));
    (cudaMalloc((void **)&d_c, N * M * sizeof(float)));

    (cudaMemcpy(d_a, h_a, N * K * sizeof(float), cudaMemcpyHostToDevice));
    (cudaMemcpy(d_b, h_b, K * M * sizeof(float), cudaMemcpyHostToDevice));
    (cudaMemcpy(d_c, h_c, N * M * sizeof(float), cudaMemcpyHostToDevice)); 

    cublasHandle_t handle; // cuBLAS context

    cublasCreate(&handle);

    auto start = std::chrono::high_resolution_clock::now();

    cublasStatus_t status = cublasSgemm(handle, 
                      CUBLAS_OP_N, CUBLAS_OP_N,  // No transpositions
                      M, N, K,                   // Dimensions (m, n, k) for column-major 
                      &alpha,                    // Alpha scaling factor
                      d_b, M,                    // Matrix B and its leading dimension
                      d_a, K,                    // Matrix A and its leading dimension
                      &beta,                     // Beta scaling factor 
                      d_c, M);                   // Result matrix C and its leading dimension
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        return 100000000000000.0f;
    }

    cudaMemcpy(h_c, d_c, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return (float)elapsed.count();
}

#endif // CUBLAS_SGEMM_H