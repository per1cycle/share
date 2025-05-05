#include <iostream>

const int N = 1024;
__global__ void MatAdd(float A[N], float B[N], float C[N])
{
    int i = threadIdx.x + blockIdx.x;
    C[i] = A[i] + B[i];
}

// TODO not implemented.
int main()
{
    std::cout << "Cuda works" << std::endl;
    float *h_a = (float*) malloc(sizeof(float) * N);
    float *h_b = (float*) malloc(sizeof(float) * N);
    float *h_c = (float*) malloc(sizeof(float) * N);
    float elapsed = 0.0f;
    int iter = 10;
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 1.0f * i;
        h_b[i] = 1.0f * i;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_c, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i = 0; i < iter; i ++)
        MatAdd<<<N / 512, 512>>>(d_A, d_B, d_C);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaMemcpy(h_c, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Cuda finished" << std::endl;
    std::cout << "elapse: " << elapsed << "ms. " << std::endl;
    std::cout << "flops: " << 1.0 * N / elapsed * iter * 1000.0f / (10000000000.f) << " gflops. " << std::endl;
    return 0;
}