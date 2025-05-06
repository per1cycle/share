#include <hip/hip_runtime.h>
#include <iostream>

class Timer
{
public:
    Timer()
    {
        hipEventCreate(&start);
        hipEventCreate(&stop);
        elapse = 0.0f;
    }

    void start()
    {
        hipEventRecord(start, 0);
    }

    void stop()
    {
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapse, start, stop);
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
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

private:
    hipEvent_t start, stop;
    float elapse;
};

namespace utils
{
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
    
void cmp_result(float *res, float *a, float *b, int M, int N, int K)
{

    for(int i = 0; i < M; i ++) // i th row
    {
        for(int j = 0; j < N; j ++) // j th column
        {
            float tmp = 0.0f;
            for(int k = 0; k < K; k ++)
            {
                tmp += a[i * K + k] * b[j + k * N];
            }

            if(abs(res[i * K + j] - tmp) > 1e-3)
            {
                std::cout << "Result error.\n";
                exit(1);
            }
        }
    }
    std::cout << "Correct! be proud of it!\n";
}

}