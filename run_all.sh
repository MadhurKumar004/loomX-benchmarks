#!/usr/bin/env bash
# run_all.sh -- run all configured benchmark suites through loomX.
#
# Usage:
#   ./run_all.sh
#
# This runs:
#   1. interproc-microbench (timing + correctness)
#   2. polybench-loomx      (timing + correctness)
#   3. autoparbench         (reference-oracle correctness; limited by loomX __float128 support)
#   4. loop-fission         (reference-oracle correctness; limited by loomX __float128 support)
#   5. dataracebench        (correctness ground truth only)
#
# Environment overrides (see config.env):
#   LOOMX, COMPILER_CPU, COMPILER_GPU, GPU_ARCH, RUNS, BENCH_ARGS, LOCK_CLOCKS, BENCH_LIMIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/config.env" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/config.env"
fi

LOOMX="${LOOMX:-$SCRIPT_DIR/../loomX/tools/loomX/loomX}"

if [ ! -x "$LOOMX" ]; then
    echo "ERROR: loomX binary not found or not executable: $LOOMX" >&2
    echo "Edit config.env or export LOOMX=/path/to/loomX" >&2
    exit 1
fi

for suite in interproc-microbench polybench-loomx autoparbench loop-fission rodinia dataracebench; do
    dir="$SCRIPT_DIR/suites/"
    case "$suite" in
        polybench) dir="${dir}polybench" ;;
        polybench-loomx) dir="${dir}polybench-loomx" ;;
        rodinia)   dir="${dir}rodinia" ;;
        dataracebench) dir="${dir}dataracebench" ;;
        autoparbench) dir="${dir}AutoParBench" ;;
        loop-fission) dir="${dir}Loop-Fission" ;;
    esac
    if [ "$suite" != "interproc-microbench" ] && [ ! -d "$dir" ]; then
        echo "WARNING: $suite not found. Run ./setup.sh first." >&2
        continue
    fi

    echo ""
    echo "##########################################################################"
    echo "# Running suite: $suite"
    echo "##########################################################################"
    "$SCRIPT_DIR/run_suite.sh" "$suite" || true
done

echo ""
echo "== All suites finished. Results are in results/ =="
ls -1 "$SCRIPT_DIR/results/" 2>/dev/null || true
