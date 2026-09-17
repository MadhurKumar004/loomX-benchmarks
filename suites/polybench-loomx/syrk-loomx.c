/* syrk-loomx.c: loomX-parseable version of PolyBench/C syrk.
 * C := alpha*A*A**T + beta*C, A is NxM, C is NxN symmetric.
 */
#include "bench_minimal.h"

#define N 1200
#define M 1000

static double C[N][N];
static double A[N][M];

int main(int argc, char** argv) {
    int i, j, k;
    double alpha = 1.5;
    double beta = 1.2;
    double sum = 0.0;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            A[i][j] = (double)((i * j + 1) % N) / N;
        }
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = (double)((i * j + 2) % M) / M;
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
            C[i][j] *= beta;
        }
        for (k = 0; k < M; k++) {
            for (j = 0; j <= i; j++) {
                C[i][j] += alpha * A[i][k] * A[j][k];
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
