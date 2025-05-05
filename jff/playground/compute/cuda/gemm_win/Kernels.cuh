#ifndef KERNEL_CUH
#define KERNEL_CUH
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <iostream>

using uint = unsigned int;

void generate_float_matrix(float* out, int row, int column)
{
    srand(time(NULL));
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            out[i * column + j] = (float)rand() / (float)(INT32_MAX);
        }
    }
}

void print_array(float* arr, int row, int column)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            std::cout << arr[i * column + j] << ' ';
        }
        std::cout << std::endl;
    }
}

void cmp_result(float* res, float* a, float* b, int M, int N, int K)
{
    float* tmp = new float[M * N];
    memset(tmp, 0, sizeof(float) * M * N);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < K; k++)
            {
                tmp[i * N + j] += a[i * K + k] * b[j + k * N];
            }

            if (abs(res[i * K + j] - tmp[i * K + j]) > 1e-2)
            {
                std::cout << "Result error.\n";
                exit(1);
            }
        }
    }
    std::cout << "Correct! be proud of it!\n";

}

__global__ void sgemm_naive(
    uint M, uint N, uint K,
	float* a, float* b, float* c,
    float alpha, float beta)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < M && y < N)
	{
		float sum = 0;
		for (int k = 0; k < K; k++)
		{
			sum += a[x * K + k] * b[k * N + y];
		}
		c[x * N + y] = alpha * sum + beta;
	}
}

void run_kernel(int kernel_type,
	uint M, uint N, uint K,
	float *a, float *b, float *c,
    float alpha, float beta)
{
    switch (kernel_type)
    {
    case 0: // cublas
    {
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			M, N, K,
			&alpha, a, M,
			b, K,
			&beta, c, M);
        break;
    }
    case 1:
    {
		dim3 grid_dim = dim3(M / 32, N / 32, 1);
		dim3 block_dim = dim3(32, 32, 1);
		sgemm_naive << <grid_dim, block_dim >> > (M, N, K, a, b, c, alpha, beta);
        break;
    }
    default:
        break;
    }
}
#endif // !KERNEL_CUH
