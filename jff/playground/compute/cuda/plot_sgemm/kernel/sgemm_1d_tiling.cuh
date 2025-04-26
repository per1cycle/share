#ifndef SGEMM_1D_TILING_CUH
#define SGEMM_1D_TILING_CUH
/**
 * assuming all of data can be divided by block
 * completely doesnot understand this.
 */
template <
    const int BLK_N,
    const int BLK_M, // reduce the m to optimize blkm
    const int BLK_K,
    const int THREAD_N
    >
__global__ void sgemm_1d_tiling(int N, int M, int K, float *a, float *b, float *c, float alpha, float beta)
{
    assert(BLK_N * BLK_M == blockDim.x);
    assert(BLK_M * BLK_K == blockDim.x);
    int c_row = blockIdx.y;
    int c_col = blockIdx.x;

    int thread_row = threadIdx.x / BLK_K;
    int thread_col = threadIdx.x % BLK_K;

    int inner_a_row = threadIdx.x / BLK_M;
    int inner_a_col = threadIdx.x % BLK_M;

    int inner_b_row = threadIdx.x / BLK_K;
    int inner_b_col = threadIdx.x % BLK_K;

    a += c_row * BLK_N * M;
    b += c_col * BLK_K;
    c += c_row * K * BLK_N + c_col * BLK_K;

    __shared__ float a_share[BLK_N * BLK_M];
    __shared__ float b_share[BLK_M * BLK_K];

    float temp_arr[THREAD_N] = {0.0f};

    /**
     * A: N * M
     * B: M * K
     */
    for(int k = 0; k < M; k += BLK_M)
    {
        a_share[inner_a_row * BLK_M + inner_a_col] = a[inner_a_row * M + inner_a_col];
        b_share[inner_b_row * BLK_K + inner_b_col] = b[inner_b_row * K + inner_b_col];

        __syncthreads();

        a += BLK_M;
        b += BLK_M * K;

        for(int inner = 0; inner < BLK_M; inner ++)
        {
            float temp_b = b_share[inner * BLK_K + thread_col];
            for(int tid = 0; tid < THREAD_N; tid++)
            {
                temp_arr[tid] += 
                    a_share[(thread_row * THREAD_N + tid) * BLK_M + inner]
                    * temp_b;
            }
        }

        __syncthreads();

    }

    for(int i = 0; i < THREAD_N; i ++)
        c[(thread_row * THREAD_N + i) * K + thread_col] = alpha * temp_arr[i] + beta;
}

#endif