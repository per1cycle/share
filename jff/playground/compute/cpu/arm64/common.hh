#ifndef COMPUTE_COMMON_CUH
#define COMPUTE_COMMON_CUH

// Define platform
#define CPU


// include header
#ifdef CUDA
#include <cuda_runtime.h>
#elif defined(HIP)
#include <hip/hip_runtime.h>
#elif defined(CPU)
#include <chrono>
#endif

#include <iostream>
#include <iomanip>
#include <random>

#if defined(CUDA)
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

template<typename T>
void print_array(T *arr, int row, int column)
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

template<typename T>
void cmp_result(T *res, T *a, T *b, int M, int N, int K)
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

#ifndef CPU
/**
 * TODO: add support for quick benchmark.
 */
template<typename T>
void quick_bench_sgemm()
{
    constexpr uint M = 2048, N = 2048, K = 2048;
    T alpha = 1.0, beta = 0.0;
    Timer t;

    T *h_a = (T*)malloc(sizeof(T) * M * K);
    T *h_b = (T*)malloc(sizeof(T) * K * N);
    T *h_c = (T*)malloc(sizeof(T) * M * N);

    utils::generate_T_matrix<T>(h_a, M, K);
    utils::generate_T_matrix<T>(h_b, K, N);

    memset(h_c, 0, sizeof(T) * M * N);

    t.start();

    // blas

    t.stop();
    t.report_sgemm(M, N, K, alpha, beta);
    
}

template<typename T>
void quick_bench_sgemv()
{
    constexpr uint M = 2048, N = 2048;

}

#endif // CPU

}
#endif // COMPUTE_COMMON_CUH
