#include "ideal_omp.h"

/* The original accumulates into a file-scope global through a callee, which is
 * an ordered side effect and not something to hand to the device directly. A
 * human instead keeps the accumulation in a local and offloads it as a proper
 * OpenMP reduction, then stores it once. a[i] is i%100, so the sum is a sum of
 * small integers exactly representable in double and any reduction order gives
 * identical bits. */
static double global_state = 0.0;

int main(int argc, char **argv) {
    int n = ideal_argc_n(argc, argv, 200000);
    double acc = 0.0;

#pragma omp target data map(to:n)
    {
#pragma omp target teams distribute parallel for num_teams(IDEAL_TEAMS) thread_limit(IDEAL_THREADS) schedule(static) reduction(+:acc)
        for (int i = 0; i < n; i++) {
            acc += (double)(i % 100);
        }
    }

    global_state = acc;
    printf("global_state: %.6f\n", global_state);
    return 0;
}
