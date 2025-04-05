/**
 * @version 1
 * matrix multiplication accelerated with opencl
 * N x K matrix multiplied with K x M matrix 
 */
__kernel void v1(const int N, const int M, int K, __global float *A, __global float *B, __global float *C)
{
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);
    float tmp = 0.0;
    
    for(int k = 0; k < K; k ++)
    {
        tmp += A[K * global_row + k] * B[k * M + global_col];
    }
    C[global_row * M + global_col] = tmp;
}