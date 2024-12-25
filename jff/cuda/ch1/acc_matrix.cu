#include <random>

const int N = 4096;
const int blocks_per_grid = 100, threads_per_block = 1024;

void generate_data_1d()
{
    
}

__global__ void accelerate_1d_arr(float *arr_1d_a, float *arr_1d_b, float *arr_1d_c)
{
    int idx = threadIdx.x;
    arr_1d_c[idx] = arr_1d_a[idx] + arr_1d_b[idx];
}
// accelerate 2d add/mul
__global__ void accelerate_2d_array(float *DevPtrA, float *DevPtrB, float *DevPtrC, size_t Pitch, int Width, int Height)
{

}
int main()
{
    
    float *host_arr_1d_a;
    float *host_arr_1d_b;
    float *host_arr_1d_c;

    host_arr_1d_a = new float[N];
    host_arr_1d_b = new float[N];
    host_arr_1d_c = new float[N];

    float *device_arr_1d_a;
    float *device_arr_1d_b;
    float *device_arr_1d_c;

    cudaMalloc(&device_arr_1d_a, sizeof(float) * N);
    cudaMalloc(&device_arr_1d_b, sizeof(float) * N);
    cudaMalloc(&device_arr_1d_c, sizeof(float) * N);

    cudaMemcpy(device_arr_1d_a, host_arr_1d_a, N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_arr_1d_b, host_arr_1d_b, N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_arr_1d_c, host_arr_1d_c, N, cudaMemcpyHostToDevice);
    
    


}