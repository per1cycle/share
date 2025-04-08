#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.cuh"

/*
 * A stand-alone script to invoke & benchmark standard cuBLAS SGEMM performance
 */

int main(int argc, char *argv[]) {
    const int N = 8, M = 4, K = 2;
    float flops = 2.0f * N * M * K;

    float *h_a = new float[N * K];
    float *h_b = new float[K * M];
    float *h_c = new float[N * M];
    float *ans = new float[N * M];
    
    gen_data(h_a, h_b, h_c, N, M, K);
    
    simple_matmul(h_a, h_b, ans, N, M, K);

    float *d_a;
    float *d_b;
    float *d_c;
    float alpha = 1.0f;
    float beta = 0.0f;  // This should be 0.0f for a clean multiplication
    
    cuCheck(cudaMalloc((void **)&d_a, N * K * sizeof(float)));
    cuCheck(cudaMalloc((void **)&d_b, K * M * sizeof(float)));
    cuCheck(cudaMalloc((void **)&d_c, N * M * sizeof(float)));

    cuCheck(cudaMemcpy(d_a, h_a, N * K * sizeof(float), cudaMemcpyHostToDevice));
    cuCheck(cudaMemcpy(d_b, h_b, K * M * sizeof(float), cudaMemcpyHostToDevice));
    cuCheck(cudaMemcpy(d_c, h_c, N * M * sizeof(float), cudaMemcpyHostToDevice));
    
    // prepare cublas sgemm
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return -1;
    }

    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // The correct order is: cublasSgemm(handle, transa, transb, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc)
    // In CUBLAS, matrices are stored in column-major order, but in C/C++ we use row-major
    // So we compute C = Bt*At instead of A*B by swapping the order and dimensions
    // a = n * k
    // b = k * m
    stat = cublasSgemm(handle, 
                      CUBLAS_OP_N, CUBLAS_OP_N,  // No transpositions
                      N, M, K,                   // Dimensions (m, n, k) for column-major 
                      &alpha,                    // Alpha scaling factor
                      d_a, N,                    // Matrix B and its leading dimension
                      d_b, K,                    // Matrix A and its leading dimension
                      &beta,                     // Beta scaling factor 
                      d_c, N);                   // Result matrix C and its leading dimension
    // stat = cublasSgemm(handle, 
    //                   CUBLAS_OP_N, CUBLAS_OP_N,  // No transpositions
    //                   M, N, K,                   // Dimensions (m, n, k) for column-major 
    //                   &alpha,                    // Alpha scaling factor
    //                   d_b, M,                    // Matrix B and its leading dimension
    //                   d_a, K,                    // Matrix A and its leading dimension
    //                   &beta,                     // Beta scaling factor 
    //                   d_c, M);                   // Result matrix C and its leading dimension
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS SGEMM failed" << std::endl;
        return -1;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    et = et * 1e-3; // convert to seconds.
    std::cout << "FLOPS: " << (flops) * 1e-9 / et << " GFLOPS." << std::endl;

    cudaMemcpy(h_c, d_c, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    if(compare_result(ans, h_c, N, M)) {
        std::cout << "Results match!" << std::endl;
    }
    // print_arr(ans, N, M, "ans_in_row_major");

    print_arr(h_a, N, K, "a");
    print_arr(h_b, K, M, "b");
    print_arr(h_c, N, M, "device");

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] ans;
    
    return 0;
}