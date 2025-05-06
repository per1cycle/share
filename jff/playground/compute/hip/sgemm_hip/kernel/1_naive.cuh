template <typename T>
__global__ void kernel_naive_impl(T *a, T *b, T *c, int M, int N, int K, float alpha, float beta)
{

}

template <typename T>
void kernel_naive(T *a, T *b, T *c, int M, int N, int K, float alpha, float beta)
{
    dim3 grid_dim = {};
    dim3 block_dim = {};
    kernel_naive_impl<T><<<grid_dim, block_dim>>>(a, b, c, M, N, K, alpha, beta);
}