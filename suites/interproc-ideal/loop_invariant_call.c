#include "ideal_omp.h"

/* poly(x) does not depend on i. A human evaluates it once on the host and passes
 * the result in, so the offloaded loop does one add per element instead of four
 * multiplies. The per-element value is bit-identical to the original because the
 * same expression is evaluated on the same x.
 *
 * The checksum reduction is also kept on the device in the same kernel, which is
 * the best a hand-written OpenMP-target version can do and therefore the fair
 * ceiling for loomX's fused output. */
int main(int argc, char **argv) {
    int n = ideal_argc_n(argc, argv, 200000);
    double x = (argc > 2) ? atof(argv[2]) : 1.5;
    double p = x * x * x - 2.0 * x * x + x + 1.0;   /* hoisted */
    double checksum = 0.0;

#pragma omp target data map(to:n)
    {
#pragma omp target teams distribute parallel for num_teams(IDEAL_TEAMS) thread_limit(IDEAL_THREADS) schedule(static) reduction(+:checksum)
        for (int i = 0; i < n; i++) {
            checksum += p + (double)i;
        }
    }

    printf("checksum: %.6f\n", checksum);
    return 0;
}
