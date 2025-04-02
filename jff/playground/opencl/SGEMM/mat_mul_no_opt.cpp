#include <iostream>

/**
 * [N x K] mul [K x M]
 * @output: N x M
 */
void matmul(int N, int M, int K, float* A, float* B, float* C)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                // A i rows mul b [j] col
                // temp += a[i][k] * b[k][j]
                temp += A[i * K + k] * B[k * M + j];
            }
            // c[i][j] = temp
            C[i * M + j] = temp;
        }
    }
}

void printrow(float* m, int row, int col, int row_num)
{
    for (int i = 0; i < col; i++) {
        std::cout << m[col * row_num + i] << ' ';
    }
    std::cout << std::endl;
}

void printcol(float* m, int row, int col, int col_num)
{
    for (int i = 0; i < row; i++) {
        std::cout << m[i * col + col_num] << '\n';
    }
    std::cout << std::endl;
}

int main()
{
    int rowa = 3, cola = 4;
    int rowb = 4, colb = 5;

    float a[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    };

    float b[] = {
        2.1, 3.2, 3.4, 4.5, 5.6,
        -1.0, -2.3, -0.1, 0.7, 1.8,
        9.9, 8.8, 7.7, 6.6, 5.5,
        4.4, 3.3, 2.2, 1.1, 0.0
    };

    float c[] = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0
    };

    matmul(3, 5, 4, a, b, c);
    for (int i = 0; i < rowa; i++) {
        for (int j = 0; j < colb; j++) {
            std::cout << c[i * colb + j] << ' ';
        }
        std::cout << std::endl;
    }
}