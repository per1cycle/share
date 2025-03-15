#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// https://stackoverflow.com/questions/13408990/how-to-generate-random-float-number-in-c
void gen_matrix_fp(const int size, float ** out)
{
    for(int i = 0; i < size; i ++)
    {
        for(int j = 0; j < size; j ++)
        {
            out[i][j] = (float)rand()/(float)(RAND_MAX/1);
        }
    }
}

void gen_matrix(const int size, int ** out)
{
    for(int i = 0; i < size; i ++)
    {
        for(int j = 0; j < size; j ++)
        {
            out[i][j] = rand() % 5;
        }
    }
}

void print_mat_fp(int size, float** mat)
{
    for(int i = 0; i < size; i ++)
    {
        for(int j = 0; j < size; j ++)
        {
            printf("%.2f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_mat_int(int size, int** mat)
{
    for(int i = 0; i < size; i ++)
    {
        for(int j = 0; j < size; j ++)
        {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * measure time of matmul func,
 * though it's a little bit un precise,
 * cause it the malloc time also include.
 */
void measure_time(void (*func)(int N), int N) 
{
    clock_t start = clock();
    (*func)(N);
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;

    printf("Function costs: %.3f seconds\n", seconds);
}

void matmul_int(int size, int **mat_a, int **mat_b, int **out)
{
    for(int i = 0; i < size; i ++)
    {
        for(int j = 0; j < size; j ++)
        {
            for(int k = 0; k < size; k ++)
            {
                out[i][j] += mat_a[i][k] * mat_b[k][j];
            }
        }
    }

}

void matmul_fp32(int size, float **mat_a, float **mat_b, float **out)
{
    for(int i = 0; i < size; i ++)
    {
        for(int j = 0; j < size; j ++)
        {
            for(int k = 0; k < size; k ++)
            {
                out[i][j] += mat_a[i][k] * mat_b[k][j];
            }
        }
    }
}

void test_matmul_fp32(int N)
{
    printf("[====] Running matmul test on fp32 using N=%d\n", N);
    float **mat_a, **mat_b;
    float **out;

    mat_a = (float**)malloc(N * sizeof(float*));
    mat_b = (float**)malloc(N * sizeof(float*));
    out = (float**)malloc(N * sizeof(float*));

    for(int i = 0; i < N; i ++)
    {
        mat_a[i] = malloc(N * sizeof(float));
        mat_b[i] = malloc(N * sizeof(float));
        out[i] = malloc(N * sizeof(float));
    }

    gen_matrix_fp(N, mat_a), gen_matrix_fp(N, mat_b);
    matmul_fp32(N, mat_a, mat_b, out);

    // print_mat_fp(N, mat_a);
    // print_mat_fp(N, mat_b);
    // print_mat_fp(N, out);

    for(int i = 0; i < N; i ++)
    {
        free(mat_a[i]);
        free(mat_b[i]);
        free(out[i]);
    }

}

void test_matmul_int(int N)
{
    printf("[====] Running matmul test on int using N=%d\n", N);
    int **mat_a, **mat_b;
    int **out;

    mat_a = (int**)malloc(N * sizeof(int*));
    mat_b = (int**)malloc(N * sizeof(int*));
    out = (int**)malloc(N * sizeof(int*));

    for(int i = 0; i < N; i ++)
    {
        mat_a[i] = malloc(N * sizeof(int));
        mat_b[i] = malloc(N * sizeof(int));
        out[i] = malloc(N * sizeof(int));
    }

    gen_matrix(N, mat_a), gen_matrix(N, mat_b);
    matmul_int(N, mat_a, mat_b, out);

    // print_mat_int(N, mat_a);
    // print_mat_int(N, mat_b);
    // print_mat_int(N, out);

    for(int i = 0; i < N; i ++)
    {
        free(mat_a[i]);
        free(mat_b[i]);
        free(out[i]);
    }

}

int main()
{
    srand(time(NULL));
    measure_time(test_matmul_int, 2048);
    measure_time(test_matmul_fp32, 2048);
}