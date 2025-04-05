__kernel void add(__global int *A, __global int *B, __global int *C)
{
    int thread_id = get_global_id(0);
    C[thread_id] = A[thread_id] + B[thread_id];
}
