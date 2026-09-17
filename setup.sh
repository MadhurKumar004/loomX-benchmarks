#!/usr/bin/env bash
# setup.sh -- fetch the external benchmark suites used for loomX validation.
#
# The custom interproc-microbench corpus is already included in suites/.
# This script clones the three external suites into suites/:
#   - PolyBench/C 4.2.1
#   - Rodinia
#   - DataRaceBench

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="${1:-$SCRIPT_DIR/suites}"
mkdir -p "$SUITE_DIR"
cd "$SUITE_DIR"

echo "== Fetching external benchmark suites into $SUITE_DIR =="

echo "== PolyBench/C 4.2.1 (Pouchet/Yuki, Ohio State mirror) =="
if [ ! -d polybench ]; then
  git clone --depth 1 https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git polybench
else
  echo "  already present, skipping"
fi

echo "== Rodinia (OpenMP + CUDA source side by side) =="
if [ ! -d rodinia ]; then
  git clone --depth 1 https://github.com/yuhc/gpu-rodinia.git rodinia || \
  git clone --depth 1 https://github.com/ouankou/rodinia.git rodinia
else
  echo "  already present, skipping"
fi

echo "== DataRaceBench (LLNL correctness ground truth) =="
if [ ! -d dataracebench ]; then
  git clone --depth 1 https://github.com/LLNL/dataracebench.git dataracebench
else
  echo "  already present, skipping"
fi

cat <<'EOF'

Done. Layout:
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
