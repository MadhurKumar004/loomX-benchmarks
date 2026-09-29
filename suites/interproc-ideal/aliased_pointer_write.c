#include "ideal_omp.h"

/* Original: three passes -- fill a[i]=i, then a[i]=a[i]*2 through a pointer
 * that aliases the input (loomX must reject this for aliasing), then checksum.
 * A human does not write the middle pass at all: the scale folds into the fill,
 * so the array is written once and read once. */
int main(int argc, char **argv) {
    int n = ideal_argc_n(argc, argv, 200000);
    double checksum = 0.0;

#pragma omp target data map(to:n)
    {
#pragma omp target teams distribute parallel for num_teams(IDEAL_TEAMS) thread_limit(IDEAL_THREADS) schedule(static) reduction(+:checksum)
        for (int i = 0; i < n; i++) {
            checksum += (double)i * 2.0;
        }
    }

    printf("checksum: %.6f\n", checksum);
    return 0;
}
