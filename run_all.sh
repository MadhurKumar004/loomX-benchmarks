#!/usr/bin/env bash
# run_all.sh -- run all configured benchmark suites through loomX.
#
# Usage:
#   ./run_all.sh
#
# This runs:
#   1. interproc-microbench (timing + correctness)
#   2. polybench            (timing + correctness)
#   3. dataracebench        (correctness ground truth only)
#
# Environment overrides (see config.env):
#   LOOMX, COMPILER_CPU, COMPILER_GPU, GPU_ARCH, RUNS, BENCH_ARGS, LOCK_CLOCKS

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

for suite in interproc-microbench polybench-loomx dataracebench; do
    dir="$SCRIPT_DIR/suites/"
    case "$suite" in
        polybench) dir="${dir}polybench" ;;
        rodinia)   dir="${dir}rodinia" ;;
        dataracebench) dir="${dir}dataracebench" ;;
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
