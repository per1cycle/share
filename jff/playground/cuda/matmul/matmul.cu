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

int main()
{
    std::cout << "Cuda works" << std::endl;
    const int N = 1024, M = 1024, K = 1024;
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

    cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    auto start = std::chrono::high_resolution_clock::now();
    mat_mul<<<grid, block>>>(d_A, d_B, d_C, N, M, K);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout << "Flops: " << 2.0f * N * M * K / elapsed.count() * 1000 / 1000 / 1000 / 1000 / 1000 << " TFLOPS." << std::endl; 

    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    print_2d(C, N, M, "C");
    return 0;
}
