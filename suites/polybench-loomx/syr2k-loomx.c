/* syr2k-loomx.c: loomX-parseable version of PolyBench/C syr2k.
 * C := alpha*A*B**T + alpha*B*A**T + beta*C, A/B are NxM, C is NxN symmetric.
 */
#include "bench_minimal.h"

#define N 1200
#define M 1000

static double C[N][N];
static double A[N][M];
static double B[N][M];

int main(int argc, char** argv) {
    int i, j, k;
    double alpha = 1.5;
    double beta = 1.2;
    double sum = 0.0;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            A[i][j] = (double)((i * j + 1) % N) / N;
            B[i][j] = (double)((i * j + 2) % N) / N;
        }
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = (double)((i * j + 3) % M) / M;
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
            C[i][j] *= beta;
        }
        for (k = 0; k < M; k++) {
            for (j = 0; j <= i; j++) {
                C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];
            }
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
            sum += C[i][j];
        }
    }

    printf("checksum: %.6f\n", sum);
    return 0;
}
