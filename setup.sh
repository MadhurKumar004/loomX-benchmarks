#!/usr/bin/env bash
# setup.sh -- verify / refresh the benchmark suites used for loomX validation.
#
# Most suites are vendored inside this repository under suites/:
#   - interproc-microbench (custom, included)
#   - PolyBench/C 4.2.1
#   - Rodinia
#   - DataRaceBench
#
# The LLVM Test Suite and NPB are large, so they are kept outside suites/ by
# default. Point LLVM_TEST_SUITE_DIR / NPB_DIR at checkouts (see config.env) or
# pass --refresh-llvm-test-suite / --refresh-npb to clone them.
#
# Running this script without arguments just checks that the directories exist.
# Pass --refresh to re-clone the vendored suites from upstream (destructive).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.env" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/config.env"
fi
SUITE_DIR="${2:-$SCRIPT_DIR/suites}"
mkdir -p "$SUITE_DIR"
cd "$SUITE_DIR"

ACTION="${1:-verify}"

refresh_suite() {
    local name="$1"
    local url="$2"
    echo "Refreshing $name from $url ..."
    rm -rf "$name"
    git clone --depth 1 "$url" "$name"
    rm -rf "$name/.git"
}

LLVM_TEST_SUITE_DIR="${LLVM_TEST_SUITE_DIR:-$SCRIPT_DIR/suites/llvm-test-suite}"
NPB_DIR="${NPB_DIR:-$SCRIPT_DIR/suites/NPB3.0-omp-C}"

if [ "$ACTION" = "--refresh" ]; then
    refresh_suite polybench       https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git
    refresh_suite rodinia         https://github.com/yuhc/gpu-rodinia.git || \
    refresh_suite rodinia         https://github.com/ouankou/rodinia.git
    refresh_suite dataracebench   https://github.com/LLNL/dataracebench.git
elif [ "$ACTION" = "--refresh-llvm-test-suite" ]; then
    refresh_suite llvm-test-suite https://github.com/llvm/llvm-test-suite.git
    exit 0
elif [ "$ACTION" = "--refresh-npb" ]; then
    refresh_suite NPB3.0-omp-C    https://github.com/benchmark-subsetting/NPB3.0-omp-C.git
    exit 0
fi

missing=0
for d in interproc-microbench polybench rodinia dataracebench; do
    if [ -d "$d" ]; then
        echo "OK: $d"
    else
        echo "MISSING: $d"
        missing=1
    fi
done

if [ -d "$LLVM_TEST_SUITE_DIR" ]; then
    echo "OK: llvm-test-suite ($LLVM_TEST_SUITE_DIR)"
else
    echo "MISSING: llvm-test-suite (set LLVM_TEST_SUITE_DIR or run '$0 --refresh-llvm-test-suite')"
fi

if [ -d "$NPB_DIR" ]; then
    echo "OK: npb ($NPB_DIR)"
else
    echo "MISSING: npb (set NPB_DIR or run '$0 --refresh-npb')"
fi

if [ "$missing" -ne 0 ]; then
    echo "ERROR: some required vendored suites are missing. Run '$0 --refresh' to fetch them." >&2
    exit 1
fi

cat <<'EOF'

All suites present:
  suites/interproc-microbench/ -- custom hand-written interprocedural tests
  suites/polybench/            -- PolyBench/C 4.2.1 kernels
  suites/rodinia/              -- Rodinia OpenMP/CUDA apps
  suites/dataracebench/        -- DRB yes/no race ground-truth files
  suites/llvm-test-suite/      -- LLVM Test Suite single-source C benchmarks
    (or point LLVM_TEST_SUITE_DIR at an external checkout)
  suites/NPB3.0-omp-C/         -- NAS Parallel Benchmarks OpenMP C version
    (or point NPB_DIR at an external checkout)

Next steps:
  1. Edit config.env to point LOOMX at your loomX binary.
  2. ./run_all.sh              # run every configured suite
  3. ./run_suite.sh interproc-microbench
     ./run_suite.sh polybench
     ./run_suite.sh dataracebench
     ./run_suite.sh llvm-test-suite
     ./run_suite.sh npb
EOF
