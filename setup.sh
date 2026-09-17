#!/usr/bin/env bash
# setup.sh -- verify / refresh the benchmark suites used for loomX validation.
#
# All suites are vendored inside this repository under suites/:
#   - interproc-microbench (custom, included)
#   - PolyBench/C 4.2.1
#   - Rodinia
#   - DataRaceBench
#
# Running this script without arguments just checks that the directories exist.
# Pass --refresh to re-clone them from upstream (destructive).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

if [ "$ACTION" = "--refresh" ]; then
    refresh_suite polybench       https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git
    refresh_suite rodinia         https://github.com/yuhc/gpu-rodinia.git || \
    refresh_suite rodinia         https://github.com/ouankou/rodinia.git
    refresh_suite dataracebench   https://github.com/LLNL/dataracebench.git
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

if [ "$missing" -ne 0 ]; then
    echo "ERROR: some suites are missing. Run '$0 --refresh' to fetch them." >&2
    exit 1
fi

cat <<'EOF'

All suites present:
  suites/interproc-microbench/ -- custom hand-written interprocedural tests
  suites/polybench/            -- PolyBench/C 4.2.1 kernels
  suites/rodinia/              -- Rodinia OpenMP/CUDA apps
  suites/dataracebench/        -- DRB yes/no race ground-truth files

Next steps:
  1. Edit config.env to point LOOMX at your loomX binary.
  2. ./run_all.sh              # run every configured suite
  3. ./run_suite.sh interproc-microbench
     ./run_suite.sh polybench
     ./run_suite.sh dataracebench
EOF
