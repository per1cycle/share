#ifndef SGEMM_2D_TILING
#define SGEMM_2D_TILING


/**
 * assuming all of data can be divided by block
 * completely doesnot understand this.
 */
template <
    const int BLK_N,
    const int BLK_M, // reduce the m to optimize blkm
    const int BLK_K,
    const int THREAD_N,
    const int THREAD_K
    >
__global__ void sgemm_2d_tiling(int N, int M, int K, float *a, float *b, float *c, float alpha, float beta)
{
    int c_row = blockIdx.x;
    int c_col = blockIdx.y;

    // the total tile of the result block.
    int total_result = BLK_N * BLK_K;
    // total tile number
    int num_thread_blocktile = total_result / (THREAD_N * THREAD_K);

    assert(num_thread_blocktile == blockDim.x);

    int thread_row = threadIdx.x / (BLK_K / THREAD_K);
    int thread_col = threadIdx.x % (BLK_K / THREAD_K);

    int stride_a = num_thread_blocktile / BLK_M;
    int inner_a_row = threadIdx.x / BLK_M;
    int inner_a_col = threadIdx.x % BLK_M;

    int stride_b = num_thread_blocktile / BLK_K;
    int inner_b_row = threadIdx.x / BLK_K;
    int inner_b_col = threadIdx.x % BLK_K;

    a += c_row * BLK_N * M;
    b += c_col * BLK_K;
    c += c_row * K * BLK_N + c_col * BLK_K;

    __shared__ float a_share[BLK_N * BLK_M];
    __shared__ float b_share[BLK_M * BLK_K];

    float thread_res[THREAD_N * THREAD_K] = {0.0f};

    float register_n[THREAD_N] = {0.0f};
    float register_k[THREAD_K] = {0.0f};

    /**
     * A: N * M
     * B: M * K
     */
    for(int blk_m = 0; blk_m < M; blk_m += BLK_M)
    {
        // move to share memory cache.
        for(int offset = 0; offset < BLK_N; offset += stride_a)
            a_share[(inner_a_row + offset) * BLK_M + inner_a_col] 
                = a[(inner_a_row + offset) * M + inner_a_col];

        for(int offset = 0; offset < BLK_M; offset += stride_b)
            b_share[(inner_b_row + offset) * BLK_K + inner_b_col] 
                = b[(inner_b_row + offset) * K + inner_b_col];
        
        __syncthreads();

        a += BLK_M;
        b += BLK_M * K;
        
        // calculate teh thread result
        for(int inner_m = 0; inner_m < BLK_M; inner_m ++)
        {
            // move share memory cache to register.
            for(int register_a_idx = 0; register_a_idx < THREAD_N; register_a_idx ++)
                register_n[register_a_idx] = a_share[(thread_row * THREAD_K + register_a_idx) * BLK_M + inner_m];
            
            for(int register_b_idx = 0; register_b_idx < THREAD_K; register_b_idx ++)
                register_k[register_b_idx] = b_share[inner_m * BLK_K + thread_col * THREAD_K + register_b_idx];

            for(int tid_n = 0; tid_n < THREAD_N; tid_n++)
            {
                for(int tid_k = 0; tid_k < THREAD_K; tid_k ++)
                {
                    thread_res[tid_n * THREAD_K + tid_k] += register_n[tid_n] * register_k[tid_k];
                }
            }
        }
        __syncthreads();
    }

    // write result back to c array.
    for(int tid_n = 0; tid_n < THREAD_N; tid_n ++)
        for(int tid_k = 0; tid_k < THREAD_K; tid_k ++)
        {
            c[(thread_row * THREAD_N + tid_n) * K + thread_col * THREAD_K + tid_k]
                = alpha * thread_res[tid_n * THREAD_K + tid_k] + beta;
        }
}

#endif