/* gemm-loomx.c: loomX-parseable version of PolyBench/C gemm.
 * Avoids system headers to work around ROSE __float128 parsing issues.
 */
#include "bench_minimal.h"

#define NI 1024
#define NJ 1024
#define NK 1024

static double A[NI][NK];
static double B[NK][NJ];
static double C[NI][NJ];

static double alpha = 1.5;
static double beta = 1.2;

int main(int argc, char** argv) {
    int i, j, k;
    double sum = 0.0;

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j++) {
            C[i][j] = (double)(i * j % NI) / NI;
        }
    }
    for (i = 0; i < NI; i++) {
        for (j = 0; j < NK; j++) {
            A[i][j] = (double)(i * (j + 1) % NK) / NK;
        }
    }
    for (i = 0; i < NK; i++) {
        for (j = 0; j < NJ; j++) {
            B[i][j] = (double)(i * (j + 2) % NJ) / NJ;
        }
    }

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j++) {
            C[i][j] = beta * C[i][j];
        }
        for (k = 0; k < NK; k++) {
            for (j = 0; j < NJ; j++) {
                C[i][j] = C[i][j] + alpha * A[i][k] * B[k][j];
            }
        }
    }

    for (i = 0; i < NI; i++) {
        for (j = 0; j < NJ; j++) {
            sum = sum + C[i][j];
        }
    }

    printf("checksum: %.6f\n", sum);
    return 0;
}
