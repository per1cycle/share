#include <iostream>
#include <string>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <assert.h>

const int loop = 10;
const int shape_num = 10;
int shape[shape_num];
std::string marker = ".ov^<>s*P+xD";


void gen_shape()
{
    for(int i = 0; i < shape_num; i ++)
    {
        shape[i] = i * 256;
    }
}

std::string current_time()
{
    // Get current time
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    // Create a string stream to format the time
    std::ostringstream oss;
    oss << std::put_time(localTime, "%Y-%d-%m-%H:%M:%S");

    return oss.str();
}
void save_fig()
{
    std::cout << "plt.savefig(" << "\"" << current_time() << ".png\"" << ")" << std::endl;
}
void quick_plot()
{
    std::cout 
            << "plt.plot([1, 2, 3, 4], [1, 4, 2, 3])" << std::endl;
    return;
}

void import()
{
    std::cout 
            << "import numpy as np" << std::endl
            << "import matplotlib.pyplot as plt" << std::endl
            << "from matplotlib.pyplot import MultipleLocator" << std::endl
            << "plt.figure(figsize=(10, 6))" << std::endl
            << "plt.xlabel('Matrix Size N (for NxN matrices)')" << std::endl
            << "plt.ylabel('Performance (GFLOPs/s)')" << std::endl
            << "plt.title('SGEMM Performance Comparison')" << std::endl
            << "plt.grid(True)" << std::endl
            << "x_major_locator = MultipleLocator(256)" << std::endl
            << "plt.gca().xaxis.set_major_locator(x_major_locator)" << std::endl;

}

template <typename T>
void py_arr(T *arr, int arr_size, bool is_arg)
{
    std::cout << '[';
    for(int i = 0; i < arr_size; i ++)
    {
        std::cout << arr[i];
        if(i != arr_size - 1)
            std::cout << ',';
    }
    std::cout << ']';
    if(is_arg) 
        std::cout << ',';
}

void py_idx_arr(int arr_size)
{
    int *tmp = (int*) malloc(sizeof(int) * arr_size);
    for(int i = 0; i < arr_size; i ++) 
        tmp[i] = (i + 1);
    py_arr<int>(tmp, arr_size, false);
}

inline void py_plt_start()
{
    std::cout << "plt.plot(" << std::endl;
}

inline void py_plt_end()
{
    std::cout << ")" << std::endl;
}

void select_marker(int i, bool is_arg)
{
    assert(i < marker.size());
    std::cout << "marker='" << marker[i] << "'";
    if(is_arg)
        std::cout << ',';
}

void add_label(std::string kernel, bool is_arg)
{
    std::cout << "label=" << "\"" << kernel << "\"";
    if(is_arg)
        std::cout << ',';
}
int main()
{
    gen_shape();
    import();

    py_plt_start();
        py_arr<int>(shape, shape_num, true);
        py_arr<int>(shape, shape_num, true);
        select_marker(1, true);
        add_label("naive", false);
    py_plt_end();

    py_plt_start();
        py_arr<int>(shape, shape_num, true);
        py_arr<int>(shape, shape_num, true);
        select_marker(2, true);
        add_label("fun", false);
    py_plt_end();

    save_fig();

    return 0;

    for(int i = 0; i < shape_num; i ++)
    {
        int N = shape[i];

        float *A, *B, *C;
        A = (float*) malloc(sizeof(float) * N);
        B = (float*) malloc(sizeof(float) * N);
        C = (float*) malloc(sizeof(float) * N);

        for (int i = 0; i < N; i++)
        {
            A[i] = i;
            B[i] = i;
        }

        float *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, N * sizeof(float));
        cudaMalloc((void **)&d_B, N * sizeof(float));
        cudaMalloc((void **)&d_C, N * sizeof(float));

        cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, C, N * sizeof(float), cudaMemcpyHostToDevice);
        

        cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
    }
    return 0;
}