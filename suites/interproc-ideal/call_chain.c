#include "ideal_omp.h"

/* step_a(x) = step_b(x) - 1, step_b(x) = step_c(x*2) = 2x + 1, so step_a(x) = 2x.
 * The call chain collapses to a multiply by hand, and init+scale+checksum fuse
 * into a single pass. Every value is an exact small integer, so the reduction
 * is order-independent. */
int main(int argc, char **argv) {
    int n = ideal_argc_n(argc, argv, 200000);
    double checksum = 0.0;

#pragma omp target data map(to:n)
    {
#pragma omp target teams distribute parallel for num_teams(IDEAL_TEAMS) thread_limit(IDEAL_THREADS) schedule(static) reduction(+:checksum)
        for (int i = 0; i < n; i++) {
            checksum += 2.0 * (double)i;
        }
    }

    printf("checksum: %.6f\n", checksum);
    return 0;
}
