#include <iostream>


// device code
__global__ void pitch_init(float* pitch_ptr, size_t pitch, int width, int height)
{
    for(int r = 0; r < height; r ++)
    {
        float* row = (float*)((char*)pitch_ptr + r * pitch);
        for(int c = 0; c < width; c ++)
        {
            row[c] = 1;
        }
    }
}

int main(int argc, char const *argv[])
{
    const int w = 4096, h = 4096;
    float* pitch_ptr;
    size_t pitch = 0;

    float* result = new float[w * h * sizeof(float)];

    // https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
    cudaError_t err = cudaMallocPitch(&pitch_ptr, &pitch, w * sizeof(float), h);
    if(err != cudaSuccess) std::cout << cudaGetErrorString(err) << '\n';

    pitch_init<<<100, 128>>>(pitch_ptr, pitch, w, h);
    cudaMemcpy2D(result, w * sizeof(float), pitch_ptr, pitch, w * sizeof(float), h, cudaMemcpyDeviceToHost);

    for(int i = 0; i < w * h; i ++)
    {
        std::cout << result[i] << ' ';
        if(i && i % w == 0) std::cout << '\n';
    }

    std::cout << pitch << '\n';

    cudaFree(pitch_ptr);
    free(result);

    return 0;
}
