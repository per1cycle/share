#include <omp.h>
#include <stdio.h>

int main()
{
    // https://curc.readthedocs.io/en/latest/programming/OpenMP-C.html
    printf("Hello world from %d\n", omp_get_thread_num());
    return 0;
}