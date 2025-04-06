#include <iostream>
#include <chrono>
__global__ void mat_mul(float *a, float *b, float *c, int N, int M, int K)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp = 0.0f;
    for(int k = 0; k < K; k ++)
        tmp += a[y * K + k] * b[k * M + x];
    c[y * M + x] = tmp;
}

void load_numpy()
{
    std::cout << "import numpy as np" << std::endl;
}

void numpy_dot(std::string arr_a, std::string arr_b, std::string arr_compare)
{
    std::cout << "tmp = ";
    std::cout << arr_a << ".dot(" << arr_b << ")" << std::endl;
    std::cout << "res = np.allclose(" << "tmp" << ", " << arr_compare << ")" << std::endl;
    std::cout << "print(res)" << std::endl;
}

void print_2d(float *a, int row, int col, std::string arr_name)
{
    std::cout << arr_name << " = np.array(";
    std::cout << "[";
    for(int i = 0; i < row; i ++)
    {
        std::cout << '[';
        for(int j = 0; j < col; j ++)
        {
            std::cout << a[i * col + j] << ", ";
        }
        if(i != row - 1)
            std::cout << "]," << std::endl;
        else 
            std::cout << "]";
    }
    std::cout << "])" << std::endl;
}
void gen_py(float *A, float *B, float *C, int N, int M, int K)
{
    load_numpy();
    print_2d(A, N, K, "A");
    print_2d(B, K, M, "B");
    print_2d(C, N, M, "C");
    numpy_dot("A", "B", "C");
}

int main()
{
    // const int N = 8192, M = 8192, K = 8192;
    const int N = 2048, M = 2048, K = 2048;
    // const int N = 1024, M = 1024, K = 1024;
    // const int N = 4, M = 4, K = 4;
    float *A = new float[N * K];
    float *B = new float[K * M];
    float *C = new float[N * M];

    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < K; j ++)
        {
            A[i * K + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    for(int i = 0; i < K; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            B[i * M + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    for(int i = 0; i < N; i ++)
    {
        for(int j = 0; j < M; j ++)
        {
            C[i * M + j] = 0;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * M * sizeof(float));
    cudaMalloc((void **)&d_C, N * M * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * M * sizeof(float), cudaMemcpyHostToDevice);
    mat_mul<<<grid, block>>>(d_A, d_B, d_C, N, M, K);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout << "Flops: " << 2.0f * N * M * K / elapsed.count() * 1000 / 1000 / 1000 / 1000 / 1000 << " TFLOPS." << std::endl; 

    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // gen_py(A, B, C, N, M, K);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
