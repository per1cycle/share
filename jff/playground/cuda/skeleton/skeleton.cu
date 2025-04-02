#include <iostream>
const int N = 1024;
__global__ void MatAdd(float A[N], float B[N], float C[N])
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    std::cout << "Cuda works" << std::endl;

    MatAdd<<<4, 256>>>()
}