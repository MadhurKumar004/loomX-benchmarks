#include "ideal_omp.h"

/* triangular(k) = k + (k-1) + ... + 1 = k(k+1)/2 for k >= 0. A human replaces the
 * recursion with the closed form, which turns an inherently sequential walk into
 * one multiply and a shift per element. The values are integers and the sum is
 * exact, so the reduction order does not matter. */
int main(int argc, char **argv) {
    int n = ideal_argc_n(argc, argv, 100000);
    long checksum = 0;

#pragma omp target data map(to:n)
    {
#pragma omp target teams distribute parallel for num_teams(IDEAL_TEAMS) thread_limit(IDEAL_THREADS) schedule(static) reduction(+:checksum)
        for (int i = 0; i < n; i++) {
            long k = (long)(i % 32);
            checksum += k * (k + 1) / 2;
        }
    }

    printf("checksum: %ld\n", checksum);
    return 0;
}
