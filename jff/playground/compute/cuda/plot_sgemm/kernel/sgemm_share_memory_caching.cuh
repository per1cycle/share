#ifndef SGEMM_SHARE_MEMORY_CACHING_HH
#define SGEMM_SHARE_MEMORY_CACHING_HH
template <const int BLK>
__global__ void sgemm_share_memory_caching(int N, int M, int K, float *a, float *b, float *c, float alpha, float beta)
{
    int c_row = blockIdx.x;
    int c_col = blockIdx.y;

    a += c_row * BLK * M;
    b += c_col * BLK;
    c += c_row * K * BLK + c_col * BLK;

    int thread_row = threadIdx.x / BLK;
    int thread_col = threadIdx.x % BLK;
    float tmp = 0.0f;

    __shared__ float a_share[BLK * BLK];
    __shared__ float b_share[BLK * BLK];

    for(int k = 0; k < M; k += BLK)
    {
        a_share[thread_row * BLK + thread_col] = a[thread_row * M + thread_col];
        b_share[thread_row * BLK + thread_col] = b[thread_row * K + thread_col];

        __syncthreads();

        a += BLK;
        b += BLK * K;

        for(int inner = 0; inner < BLK; inner ++)
        {
            tmp += a_share[thread_row * BLK + inner] * b_share[inner * BLK + thread_col];
        }

        __syncthreads();

    }

    c[thread_row * K + thread_col] = alpha * tmp + beta;
}
#endif