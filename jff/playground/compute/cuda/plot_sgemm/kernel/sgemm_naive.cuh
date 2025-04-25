#ifndef SGEMM_NAIVE_CUH
#define SGEMM_NAIVE_CUH
__global__ void sgemm_naive(int N, int M, int K, float *a, float *b, float *c, float alpha, float beta)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x <= N && y <= K)
    {
        float tmp = 0.0f;
        for(int k = 0; k < M; k ++)
        {
            tmp += a[x * M + k] * b[k * K + y];
        }

        c[x * K + y] = alpha * tmp + beta;
    }
}

#endif