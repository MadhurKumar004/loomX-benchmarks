#include "ideal_omp.h"

/* even_op(x) = x*0.5 is inlined by hand. The branch is on i, not on data, so it
 * is uniform across the whole offload and the conditional stays. init+select+
 * checksum fuse into one pass. */
int main(int argc, char **argv) {
    int n = ideal_argc_n(argc, argv, 200000);
    double checksum = 0.0;

#pragma omp target data map(to:n)
    {
#pragma omp target teams distribute parallel for num_teams(IDEAL_TEAMS) thread_limit(IDEAL_THREADS) schedule(static) reduction(+:checksum)
        for (int i = 0; i < n; i++) {
            double v = (double)i;
            checksum += (i % 2 == 0) ? v * 0.5 : v;
        }
    }

    printf("checksum: %.6f\n", checksum);
    return 0;
}
