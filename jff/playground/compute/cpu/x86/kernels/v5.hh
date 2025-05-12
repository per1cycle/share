#include <immintrin.h>
#include <iostream>
namespace solution
{
void play()
{
    float fun[16];
    float res[16];

    for(int i = 0; i < 16; i ++)
    {
        fun[i] = 1.0f * i;
    }
    __m512 avx512_fun = _mm512_load_ps((void const*)fun);
    __m512 avx512_fun2 = _mm512_load_ps((void const*)fun);
    __m512 avx512_res = _mm512_add_ps(avx512_fun, avx512_fun2);
    
    _mm512_store_ps((void*)res, avx512_res);

    for(int i = 0; i < 16; i ++)
    {
        std::cout << res[i] << " " << std::endl;
    }
}
}