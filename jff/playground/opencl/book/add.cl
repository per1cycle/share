__kernel add(float *A, float *B, float *C)
{
    int thread_id = get_global_id(0);
    C[thread_id] = A[thread_id] + B[thread_id];
}