#include <iostream>
#include <random>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>


class Timer
{
public:
    Timer()
    {
        hipEventCreate(&st);
        hipEventCreate(&ed);
        elapse = 0.0f;
    }

    void start()
    {
        hipEventRecord(st, 0);
    }

    void stop()
    {
        hipEventRecord(ed, 0);
        hipEventSynchronize(ed);
        hipEventElapsedTime(&elapse, st, ed);
    }

    // get elapsed time in second
    float elapsed()
    {
        return elapse / 1000.0f; // convert to second
    }

    void reset()
    {
        elapse = 0.0f;
    }

    ~Timer()
    {
    }

private:
    hipEvent_t st, ed;
    float elapse;
};

void print_array(float *arr, int row, int column)
{
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            std::cout << arr[i * column + j] << ' ';
        }
        std::cout << std::endl;
    }
}
void generate_float_matrix(float *out, int row, int column)
{
    srand(time(NULL));
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            out[i * column + j] = (float)rand() / (float)(INT32_MAX);
        }
    }
}

int main(int argc, char ** argv)
{
    int arg = atoi(argv[1]);
    int M = arg, N = arg, K = arg;
    float flop = 1.0 * N * N * (2 * N + 1);
    float gflop = flop / 1000000000.0f;
    float alpha = 1.0f, beta = 0.0f;

    float *h_a = (float*) malloc(sizeof(float) * M * K);
    float *h_b = (float*) malloc(sizeof(float) * K * N);
    float *h_c = (float*) malloc(sizeof(float) * M * N);

    generate_float_matrix(h_a, M, K);
    generate_float_matrix(h_b, K, N);

    float *d_a;
    float *d_b;
    float *d_c;

    hipMalloc((void **)&d_a, M * K * sizeof(float));
    hipMalloc((void **)&d_b, K * N * sizeof(float));
    hipMalloc((void **)&d_c, M * N * sizeof(float));

    hipMemcpy(d_a, h_a, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, K * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_c, h_c, M * N * sizeof(float), hipMemcpyHostToDevice);

    rocblas_handle handle;
    rocblas_create_handle(&handle);
    Timer t;

    t.start();

    rocblas_sgemm(
        handle,
        rocblas_operation_none, // Transpose option for A
        rocblas_operation_none, // Transpose option for B
        M,                      // Number of rows in A and C
        N,                      // Number of columns in B and C
        K,                      // Number of columns in A and rows in B
        &alpha,                 // alpha
        d_a,                    // Matrix A on the device
        M,                      // Leading dimension of A
        d_b,                    // Matrix B on the device
        K,                      // Leading dimension of B
        &beta,                  // beta
        d_c,                    // Matrix C on the device
        M                       // Leading dimension of C
        );
    t.stop();

    std::cout 
            << "Time:                                   \t" << t.elapsed() << "ms.\n"
            << "GFlop:                                  \t" << gflop << "\n"
            << "GFLOPS:                                 \t" << gflop / t.elapsed() << "\n";
    hipMemcpy(h_c, d_c, M * N * sizeof(float), hipMemcpyDeviceToHost);
    print_array(h_c, M, N);
    return 0;
}