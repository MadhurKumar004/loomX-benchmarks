#include "ideal_omp.h"

/* The operator is chosen on the host from n, which is known before the loop, so
 * a human devirtualises it: the call becomes two constants applied to i, and the
 * indirect call and its per-element branch both disappear. */
int main(int argc, char **argv) {
    int n = ideal_argc_n(argc, argv, 200000);
    double checksum = 0.0;

    int use_scale = (n % 2 == 0);
    double k = use_scale ? 3.0 : 1.0;   /* scale: x*3, offset: x+7 */
    double c = use_scale ? 0.0 : 7.0;

#pragma omp target data map(to:n)
    {
#pragma omp target teams distribute parallel for num_teams(IDEAL_TEAMS) thread_limit(IDEAL_THREADS) schedule(static) reduction(+:checksum)
        for (int i = 0; i < n; i++) {
            checksum += k * (double)i + c;
        }
    }

    printf("checksum: %.6f\n", checksum);
    return 0;
}
