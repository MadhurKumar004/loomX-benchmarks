#include "ideal_omp.h"

/* scale(x) = 2x + 1 inlined by hand; init+scale+checksum fuse into one pass.
 * All values are exact integers, so the reduction is order-independent. */
int main(int argc, char **argv) {
    int n = ideal_argc_n(argc, argv, 200000);
    double checksum = 0.0;

#pragma omp target data map(to:n)
    {
#pragma omp target teams distribute parallel for num_teams(IDEAL_TEAMS) thread_limit(IDEAL_THREADS) schedule(static) reduction(+:checksum)
        for (int i = 0; i < n; i++) {
            checksum += 2.0 * (double)i + 1.0;
        }
    }

    printf("checksum: %.6f\n", checksum);
    return 0;
}
