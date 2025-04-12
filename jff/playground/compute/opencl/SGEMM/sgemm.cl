/**
 * @version 1
 * matrix multiplication accelerated with opencl
 * N x K matrix multiplied with K x M matrix 
 */
__kernel void v1(const int N, const int M, int K, __global float *A, __global float *B, __global float *C)
{
    const int global_row = get_global_id(1);
    const int global_col = get_global_id(0);
    float tmp = 0.0f;
    float foo = 0.0f;
    for(int k = 0; k < K; k ++)
    {
        // AtomicAdd(foo, 1.0f); doesnot have built in intrinsic for float point atomic add.
        tmp += A[K * global_row + k] * B[k * M + global_col];
    }
    C[global_row * M + global_col] = tmp;
}

/**
 * @version 2
 * matrix multiplication accelerated with opencl
 * N x K matrix multiplied with K x M matrix 
 * tiling
 */
__kernel void v2(const int N, const int M, const int K,
                 __global float *A, __global float *B, __global float *C)
{
    const int TS = 32;
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int global_row = get_group_id(1) * TS + row;
    const int global_col = get_group_id(0) * TS + col;

    float tmp = 0.0f;

    __local float t_a[TS][TS];
    __local float t_b[TS][TS];

    for (int i = 0; i < (K) / TS; i++) {
        int tiled_col = i * TS + col;
        int tiled_row = i * TS + row;

        // Load tile from A and B into local memory
        t_a[row][col] = A[global_row * K + tiled_col];
        t_b[row][col] = B[tiled_row * M + global_col];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            tmp += t_a[row][k] * t_b[k][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[global_row * M + global_col] = tmp;
}
