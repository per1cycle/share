#ifndef COMPUTE_COMMON_CUH
#define COMPUTE_COMMON_CUH

// Define platform
#define CUDA


// include header
#ifdef CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#elif defined(HIP)
#include <hip/hip_runtime.h>
#elif defined(CPU)
#endif

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <thread>

#if defined(CUDA)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / y)
#define Event_t cudaEvent_t
#define EventCreate(event) cudaEventCreate((event))
#define EventRecord(event, value) cudaEventRecord((event), (value))
#define EventSynchronize(event) cudaEventSynchronize((event))
#define EventElapsedTime(elapse, start, end) cudaEventElapsedTime((elapse), (start), (end)) 
#define EventDestroy(event) cudaEventDestroy((event))
#define Malloc()
#define Free()
#define Memcpy()
#define D2H 0
#define H2D 1
#elif defined(HIP)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / y)
#define Event_t hipEvent_t
#define EventCreate(event) hipEventCreate ((event))
#define EventRecord(event, value) hipEventRecord((event), (value))
#define EventSynchronize(event) hipEventSynchronize((event))
#define EventElapsedTime(elapse, start, end) hipEventElapsedTime((elapse), (start), (end)) 
#define EventDestroy(event) hipEventDestroy((event))
#define Malloc()
#define Free()
#define Memcpy()
#define D2H 0
#define H2D 1
#elif defined(CPU)
#define Event_t std::chrono::system_clock::time_point
#define EventCreate(event) 
#define EventRecord(event, value) (event) = std::chrono::system_clock::now()
#define EventSynchronize(event)
#define EventElapsedTime(elapse, start, end) *(elapse) = std::chrono::duration_cast<std::chrono::milliseconds>((end) - (start)).count()
#define EventDestroy(event)
#endif

class Timer
{
public:
    Timer()
    {
        EventCreate(&start_);
        EventCreate(&stop_);
        elapse_in_milisecond_ = 0.0f;
    }

    void start()
    {
        EventRecord(start_, 0);
    }

    void stop()
    {
        EventRecord(stop_, 0);
        EventSynchronize(stop_);
        EventElapsedTime(&elapse_in_milisecond_, start_, stop_);
    }

    // get elapsed time in second
    float elapse_in_milisecond()
    {
        return elapse_in_milisecond_; // convert to second
    }

    float elapse_in_second()
    {
        return elapse_in_milisecond_ / 1000.0f; // convert to second
    }

    void reset()
    {
        elapse_in_milisecond_ = 0.0f;
    }

    void sleep_with_seconds(int seconds)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(seconds * 1000));
    }

    void just_report_time()
    {
        std::cout 
                << ">> Problem: Just report    \t" << std::endl
                << "Time elapse(in second):    \t" << elapse_in_second() << "s."<< std::endl
                << "Time elapse(in milisecond):\t" << elapse_in_milisecond() << "ms." << std::endl;
    }

    /**
     * report the information of problem.
     * alpha and beta is ignore, but still count.
     */
    void report_sgemm(uint M, uint N, uint K, [[maybe_unused]] float alpha, [[maybe_unused]] float beta)
    {
        // alpha and beta is unused.
        float flop = 2.0f * M * N * K;
        float mflop = flop / 1000000.0f;
        float gflop = mflop / 1000.0f;

        std::cout 
                << ">> Problem: SGEMM          \t" << std::endl
                << "Problem size:              \t" << "M = " << M << ", N = " << N << ", K = " << K << std::endl 
                << "Flop:                      \t" << flop << std::endl
                << "mFlop:                     \t" << mflop << std::endl
                << "gflop:                     \t" << gflop << std::endl
                << "Time elapse(in second):    \t" << elapse_in_second() << "s."<< std::endl
                << "Time elapse(in milisecond):\t" << elapse_in_milisecond() << "ms." << std::endl
                << "MFlops:                    \t" << mflop / elapse_in_second() << std::endl
                << "GFlops:                    \t" << gflop / elapse_in_second() << std::endl;
    }

    void report_sgemm_with_loop(uint M, uint N, uint K, [[maybe_unused]] float alpha, [[maybe_unused]] float beta, int loop_time)
    {
        // alpha and beta is unused.
        float flop = 2.0f * M * N * K;
        float mflop = flop / 1000000.0f;
        float gflop = mflop / 1000.0f;
        float f_loop_time = 1.0f * loop_time;

        std::cout 
                << ">> Problem: SGEMM                \t" << std::endl
                << "Loop time:                       \t" << "[" << loop_time << "]" << std::endl
                << "Problem size:                    \t" << "M = " << M << ", N = " << N << ", K = " << K << std::endl 
                << "Flop:                            \t" << flop << std::endl
                << "mFlop:                           \t" << mflop << std::endl
                << "gflop:                           \t" << gflop << std::endl
                << "loop time:                       \t" << loop_time << std::endl
                << "Time elapse(in second total):    \t" << elapse_in_second() << "s."<< std::endl
                << "Time elapse(in milisecond total):\t" << elapse_in_milisecond() << "ms." << std::endl
                << "Time elapse(in second avg):      \t" << elapse_in_second() / f_loop_time << "s."<< std::endl
                << "Time elapse(in milisecond avg):  \t" << elapse_in_milisecond() / f_loop_time << "ms." << std::endl
                << "MFlops:                          \t" << mflop / (elapse_in_second() / f_loop_time) << std::endl
                << "GFlops:                          \t" << gflop / (elapse_in_second() / f_loop_time) << std::endl;
    }

    void report_sgemv(uint M, uint N, [[maybe_unused]] float alpha, [[maybe_unused]] float beta)
    {
        // alpha and beta is unused.
        float flop = 2.0f * M * N; // TODO
        float mflop = flop / 1000000.0f;
        float gflop = mflop / 1000.0f;

        std::cout 
                << ">> Problem: SGEMV          \t" << std::endl
                << "Problem size:              \t" << "M = " << M << ", N = " << N << std::endl 
                << "Flop:                      \t" << flop << std::endl
                << "mFlop:                     \t" << mflop << std::endl
                << "gflop:                     \t" << gflop << std::endl
                << "Time elapse(in second):    \t" << elapse_in_second() << "s."<< std::endl
                << "Time elapse(in milisecond):\t" << elapse_in_milisecond() << "ms." << std::endl
                << "MFlops:                    \t" << mflop / elapse_in_second() << std::endl
                << "GFlops:                    \t" << gflop / elapse_in_second() << std::endl;
    }

    void report_sgemv_with_loop(uint M, uint N, [[maybe_unused]] float alpha, [[maybe_unused]] float beta, int loop_time)
    {
        // alpha and beta is unused.
        float flop = 2.0f * M * N; // TODO
        float mflop = flop / 1000000.0f;
        float gflop = mflop / 1000.0f;
        float f_loop_time = 1.0f * loop_time;

        std::cout 
                << ">> Problem: SGEMV                \t" << std::endl
                << "Loop time:                       \t" << "[" << loop_time << "]" << std::endl
                << "Problem size:                    \t" << "M = " << M << ", N = " << N << std::endl 
                << "Flop:                            \t" << flop << std::endl
                << "mFlop:                           \t" << mflop << std::endl
                << "gflop:                           \t" << gflop << std::endl
                << "Time elapse(in second total):    \t" << elapse_in_second() << "s."<< std::endl
                << "Time elapse(in milisecond total):\t" << elapse_in_milisecond() << "ms." << std::endl
                << "Time elapse(in second avg):      \t" << elapse_in_second() / f_loop_time << "s."<< std::endl
                << "Time elapse(in milisecond avg):  \t" << elapse_in_milisecond() / f_loop_time << "ms." << std::endl
                << "MFlops:                          \t" << mflop / (elapse_in_second() / f_loop_time) << std::endl
                << "GFlops:                          \t" << gflop / (elapse_in_second() / f_loop_time) << std::endl;
    }

    ~Timer()
    {
        EventDestroy(start_);
        EventDestroy(stop_);
    }
private:
    Event_t start_, stop_;
    float elapse_in_milisecond_;
};

namespace utils
{

namespace gpu // TODO
{

/**
 * TODO: add support for quick benchmark.
 */
template<typename T>
void quick_bench_sgemm()
{
}

template <typename T>
void quick_bench_sgemv()
{
}

template<typename T>
void quick_bench_memory_bandwidth()
{
}

template<typename T>
__global__ void sgemm_generate_reference(uint M, uint N, uint K, T *a, T *b, T *c, T alpha, T beta)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < M && y < N)
    {
        float tmp = 0;
        for(int i = 0; i < K; i ++)
            tmp += a[x * K + i] * b[i * N + y];
        c[x * N + y] = alpha * tmp + beta * c[x * N + y];
    }
}

#if defined(CUDA) || defined(HIP)
void generate_half_matrix(half *out, int row, int column)
{
    srand(time(NULL));
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            out[i * column + j] = __float2half((float)rand() / (float)(INT32_MAX));
        }
    }
}

void print_half_array_row_major(half *arr, int row, int column)
{
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            std::cout << std::setw(8) << std::setprecision(5) << __half2float(arr[i * column + j]) << ' ';
        }
        std::cout << std::endl;
    }
}

#endif

}

namespace cpu
{
template<typename T>
void transpose(T *out, T *in, const int row, const int column)
{
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            out[j * row + i] = in[i * column + j];
        }
    }
}

template<typename T>
void sgemm_T_compute(T *out, T *a, T *b, T *c, int M, int N, int K, float alpha, float beta)
{
    for(int i = 0; i < M; i ++)
    {
        for(int j = 0; j < N; j ++)
        {
            T tmp = 0;
            for(int k = 0; k < K; k ++)
            {
                tmp += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = alpha * tmp + beta * c[i * N + j];
        }
    }
}


template<typename T>
void cpu_sgemv_compute(T *out, T *a, T *x, T *y, int M, int N, float alpha, float beta)
{
    for(int i = 0; i < M; i ++)
    {
        T tmp = 0;

        for(int j = 0; j < N; j ++)
        {
            tmp += a[i * N + j] * x[j];
        }
        out[i] = alpha * tmp + beta * y[i];
    }
}

#if defined(CUDA) || defined(HIP)
void sgemm_half(uint M, uint N, uint K, half *a, half *b, half *c, half alpha, half beta)
{
    for(int i = 0; i < M; i ++)
    {
        for(int j = 0; j < N; j ++)
        {
            float tmp = 0.0f;
            for(int k = 0; k < K; k ++)
            {
                tmp += __half2float(a[i * K + k]) * __half2float(b[k * N + j]);
            }
            c[i * N + j] = __float2half(tmp);
        }
    }
}

void sgemm_half_no_alpha_and_beta(uint M, uint N, uint K, half *a, half *b, half *c)
{
    sgemm_half(M, N, K, a, b, c, __float2half(1.0f), __float2half(0.0f));
}

void sgemm_half_validate(uint M, uint N, half *ref, half *src)
{
    int num_error = 0;
    for(int i = 0; i < M; i ++)
    {
        for(int j = 0; j < N; j ++)
        {
            int idx = i * N + j;
            if(std::abs(
                __half2float(ref[idx]) - 
                __half2float(src[idx])
            ) > 1e-3)
            {
                num_error ++;
                std::cout << "Error at: " << "(" << i << ", " << j << ")" << std::endl;
            }
        }
    }
    if(num_error)
    {
        std::cout << "Total error: " << num_error << std::endl;
    }
    else 
    {
        std::cout << "Correct! be proud of it." << std::endl;
    }
    
}

#endif

}

/**
 * @brief This function generate a column major matrix, for example
 * generate_T_matrix_column_major<float>(a, 2, 3)
 * the matrix mem layout looks like this:
 * left(index)                 right(value)
 * [                            [
 * -----> N
 * | 0,       1,        2,       0.1,        0.99,      0.23,
 * v 3        4,        5,       0.12,       0.75,      0.01
 * M]                            ]
 */
template<typename T>
void generate_T_matrix(T *out, int row, int column)
{
    srand(time(NULL));
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            out[i * column + j] = (T)rand() / (T)(INT32_MAX);
        }
    }
}

#if defined(CUDA) || defined(HIP)
void generate_half_matrix(half *out, int row, int column)
{
    srand(time(NULL));
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            out[i * column + j] = __float2half((float)rand() / (float)(INT32_MAX));
        }
    }
}
#endif 

// /**
//  * @brief This function generate a column major matrix, for example
//  * generate_T_matrix_column_major<float>(a, 2, 3)
//  * the matrix mem layout looks like this:
//  * left(index)                 right(value)
//  * [                            [
//  * -----> M
//  * | 0,       3,                  0.1,        0.99,
//  * | 1,       4,                  0.42,       0.02
//  * v 2        5,                  0.12,       0.75
//  * N]                            ]
//  */
// template<typename T>
// void generate_T_matrix(T *out, int row, int column)
// {
//     srand(time(NULL));
//     for(int j = 0; j < column; j ++)
//     {
//         for(int i = 0; i < row; i ++)
//         {
//             out[i + j * row] = (T)rand() / (T)(INT32_MAX);
//         }
//     }
// }

template<typename T>
void print_array_row_major(T *arr, int row, int column)
{
    for(int i = 0; i < row; i ++)
    {
        for(int j = 0; j < column; j ++)
        {
            std::cout << std::setw(8) << std::setprecision(5) << arr[i * column + j] << ' ';
        }
        std::cout << std::endl;
    }
}

template<typename T>
void print_array_column_major(T *arr, int row, int column)
{
    for(int j = 0; j < column; j ++)
    {
        for(int i = 0; i < row; i ++)
        {
            std::cout << std::setw(8) << std::setprecision(5) << arr[i + j * row] << ' ';
        }
        std::cout << std::endl;
    }
}


template<typename T>
void sgemm_validate_result(T *res, T *a, T *b, T *c, int M, int N, int K, float alpha, float beta)
{
    for(int i = 0; i < M; i ++) // i th row
    {
        for(int j = 0; j < N; j ++) // j th column
        {
            T tmp = 0.0f;
            for(int k = 0; k < K; k ++)
            {
                tmp += a[i * K + k] * b[j + k * N];
            }
            tmp = alpha * tmp + beta * c[i * M + j];

            if(abs(res[i * K + j] - tmp) > 1e-3)
            {
                std::cout 
                        << "Result error." << std::endl
                        << "At:        \t" << "<" << i << ", " << j << ">" << std::endl
                        << "Should be: \t" << tmp << std::endl
                        << "But got:   \t" << res[i * K + j] << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Correct! be proud of it!\n";
}

template<typename T>
void sgemv_validate_result(T *res, T *a, T *x, T *y, int M, int N, float alpha, float beta)
{
    for(int i = 0; i < M; i ++) // i th row
    {
        T tmp = 0.0f;
        for(int j = 0; j < N; j ++) // j th column
        {
            tmp += a[i * N + j] * x[j];
        }

        tmp = alpha * tmp + beta * y[i];

        if(abs(res[i] - tmp) > 1e-3)
        {
            std::cout 
                    << "Result error." << std::endl
                    << "At:        \t" << "<" << i << ">" << std::endl
                    << "Should be: \t" << tmp << std::endl
                    << "But got:   \t" << res[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "Correct! be proud of it!\n";
}

}
#endif // COMPUTE_COMMON_CUH
