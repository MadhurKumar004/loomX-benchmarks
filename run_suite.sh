#!/usr/bin/env bash
# run_suite.sh -- run one benchmark suite through loomX.
#
# Usage:
#   ./run_suite.sh [interproc-microbench|polybench|dataracebench]
#
# Environment overrides (see config.env):
#   LOOMX, COMPILER_CPU, COMPILER_GPU, GPU_ARCH, RUNS, BENCH_ARGS, LOCK_CLOCKS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load defaults; user can override by exporting beforehand.
if [ -f "$SCRIPT_DIR/config.env" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/config.env"
fi

SUITE="${1:-interproc-microbench}"
LOOMX="${LOOMX:-$SCRIPT_DIR/../loomX/tools/loomX/loomX}"
COMPILER_CPU="${COMPILER_CPU:-gcc}"
COMPILER_GPU="${COMPILER_GPU:-clang}"
RUNS="${RUNS:-10}"
LOCK_CLOCKS="${LOCK_CLOCKS:-no}"

# Auto-detect GPU compute capability if not provided.
if [ -z "${GPU_ARCH:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
        GPU_ARCH="sm_${CAP}"
    else
        GPU_ARCH="sm_80"
    fi
fi

mkdir -p out bin results

# ---------------------------------------------------------------------------
# DataRaceBench is a correctness-only pass.
# ---------------------------------------------------------------------------
if [ "$SUITE" = "dataracebench" ]; then
    DRB_DIR="$SCRIPT_DIR/suites/dataracebench"
    if [ ! -d "$DRB_DIR" ]; then
        echo "ERROR: $DRB_DIR not found. Run ./setup.sh first." >&2
        exit 1
    fi
    echo "== DataRaceBench correctness pass =="
    python3 "$SCRIPT_DIR/scripts/check_dataracebench.py" \
        --loomx "$LOOMX" \
        --suite "$DRB_DIR/micro-benchmarks" \
        --mode cpu-only \
        --output "$SCRIPT_DIR/results/dataracebench.csv"
    exit 0
fi

# ---------------------------------------------------------------------------
# Pick the benchmark list and source directory.
# ---------------------------------------------------------------------------
declare -a BENCHES
if [ "$SUITE" = "interproc-microbench" ]; then
    MICRO_DIR="$SCRIPT_DIR/suites/interproc-microbench"
    if [ ! -d "$MICRO_DIR" ]; then
        echo "ERROR: $MICRO_DIR not found." >&2
        exit 1
    fi
    mapfile -t BENCHES < <(find "$MICRO_DIR" -maxdepth 1 -name '*.c' -printf '%f\n' | sed 's/\.c$//' | sort)
    BENCH_SRC_DIR="$MICRO_DIR"
    BENCH_ARGS="${BENCH_ARGS:-200000}"
elif [ "$SUITE" = "polybench" ]; then
    PB_DIR="$SCRIPT_DIR/suites/polybench"
    if [ ! -d "$PB_DIR" ]; then
        echo "ERROR: $PB_DIR not found. Run ./setup.sh first." >&2
        exit 1
    fi
    BENCHES=(gemm syrk syr2k)
    BENCH_SRC_DIR="$PB_DIR/linear-algebra/blas"
    PB_UTILITIES_DIR="$PB_DIR/utilities"
    BENCH_ARGS="${BENCH_ARGS:-}"
else
    echo "ERROR: unknown suite '$SUITE'" >&2
    exit 1
fi

echo "== loomX benchmark driver =="
echo "suite:        $SUITE"
echo "LOOMX:        $LOOMX"
echo "CPU compiler: $COMPILER_CPU"
echo "GPU compiler: $COMPILER_GPU"
echo "GPU arch:     $GPU_ARCH"
echo "runs:         $RUNS"
echo "args:         ${BENCH_ARGS:-<none>}"
echo "lock clocks:  $LOCK_CLOCKS"
echo

if [ ! -x "$LOOMX" ]; then
    echo "ERROR: loomX binary not found or not executable: $LOOMX" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Generate variants.
# ---------------------------------------------------------------------------
generate_one() {
    local name="$1"
    local src="$2"
    mkdir -p "out/$name"

    cp "$src" "out/$name/${name}__seq.c"

    local loomx_args=()
    if [ "$SUITE" = "polybench" ]; then
        loomx_args+=(-I"$PB_UTILITIES_DIR" -I"$BENCH_SRC_DIR/$name")
    fi

    "$LOOMX" --cpu-only "${loomx_args[@]}" "$src" -o "out/$name/${name}__cpu_omp.c" >/dev/null 2>&1 || {
        echo "  WARN: loomX --cpu-only failed for $name"
        return 1
    }
    "$LOOMX" --gpu-naive "${loomx_args[@]}" "$src" -o "out/$name/${name}__gpu_naive.c" >/dev/null 2>&1 || {
        echo "  WARN: loomX --gpu-naive failed for $name"
        return 1
    }
    "$LOOMX" --gpu-profitable "${loomx_args[@]}" "$src" -o "out/$name/${name}__gpu_profitable.c" >/dev/null 2>&1 || {
        echo "  WARN: loomX --gpu-profitable failed for $name"
        return 1
    }
    return 0
}

echo "== Generating source variants =="
for name in "${BENCHES[@]}"; do
    if [ "$SUITE" = "polybench" ]; then
        src="$BENCH_SRC_DIR/$name/${name}.c"
    else
        src="$BENCH_SRC_DIR/${name}.c"
    fi
    echo "  $name"
    generate_one "$name" "$src" || true
done

# ---------------------------------------------------------------------------
# Compile variants.
# ---------------------------------------------------------------------------
compile_one() {
    local name="$1"
    local cfg="$2"
    local src="out/$name/${name}__${cfg}.c"
    local bin="bin/${name}__${cfg}"
    [ -f "$src" ] || return 1

    local extra_flags=(-lm)
    if [ "$SUITE" = "polybench" ]; then
        extra_flags+=(-DPOLYBENCH_TIME -DLARGE_DATASET -I"$PB_UTILITIES_DIR" -I"$BENCH_SRC_DIR/$name")
        extra_flags+=("$PB_UTILITIES_DIR/polybench.c")
    elif [ "$SUITE" = "interproc-microbench" ]; then
        extra_flags+=(-I"$BENCH_SRC_DIR")
    fi

    case "$cfg" in
        seq)
            "$COMPILER_CPU" -O3 "$src" "${extra_flags[@]}" -o "$bin"
            ;;
        cpu_omp)
            "$COMPILER_CPU" -O3 -fopenmp "$src" "${extra_flags[@]}" -o "$bin"
            ;;
        gpu_naive|gpu_profitable)
            local omp_include="$(dirname "$COMPILER_GPU")/../projects/openmp/runtime/src"
            [ -f "$omp_include/omp.h" ] || omp_include="$(dirname "$COMPILER_GPU")/../include"
            "$COMPILER_GPU" -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
                -Xopenmp-target -march="$GPU_ARCH" \
                -I"$omp_include" \
                "$src" "${extra_flags[@]}" -o "$bin"
            ;;
    esac
}

echo "== Compiling =="
for name in "${BENCHES[@]}"; do
    for cfg in seq cpu_omp gpu_naive gpu_profitable; do
        if compile_one "$name" "$cfg"; then
            echo "  ok  $name/$cfg"
        else
            echo "  skip $name/$cfg (source or compile failed)"
        fi
    done
done

# ---------------------------------------------------------------------------
# Correctness checks.
# ---------------------------------------------------------------------------
RESULTS_CSV="results/${SUITE}.csv"
rm -f "$RESULTS_CSV"

echo "== Correctness checks =="
for name in "${BENCHES[@]}"; do
    seq_bin="bin/${name}__seq"
    [ -x "$seq_bin" ] || continue

    golden="results/${name}__golden.out"
    ./"$seq_bin" $BENCH_ARGS > "$golden" 2>/dev/null || true

    for cfg in cpu_omp gpu_naive gpu_profitable; do
        cand="bin/${name}__${cfg}"
        [ -x "$cand" ] || continue
        cand_out="results/${name}__${cfg}.out"
        ./"$cand" $BENCH_ARGS > "$cand_out" 2>/dev/null || true
        if python3 "$SCRIPT_DIR/scripts/check_correctness.py" "$golden" "$cand_out" --rtol 1e-5 --atol 1e-8; then
            echo "  PASS $name/$cfg"
        else
            echo "  FAIL $name/$cfg"
        fi
    done
done

# ---------------------------------------------------------------------------
# Timing harness.
# ---------------------------------------------------------------------------
[ "$LOCK_CLOCKS" = "yes" ] && "$SCRIPT_DIR/scripts/lock_gpu_clocks.sh" lock

echo "== Timing ($RUNS runs each) =="
for name in "${BENCHES[@]}"; do
    for cfg in seq cpu_omp gpu_naive gpu_profitable; do
        bin="bin/${name}__${cfg}"
        [ -x "$bin" ] || continue
        extra=""
        [[ "$cfg" == gpu_* ]] && extra="--nsys"
        python3 "$SCRIPT_DIR/scripts/bench_harness.py" \
            --binary "$bin" --args "$BENCH_ARGS" --runs "$RUNS" \
            --label "${name}__${cfg}" --out "$RESULTS_CSV" $extra || true
    done
done

[ "$LOCK_CLOCKS" = "yes" ] && "$SCRIPT_DIR/scripts/lock_gpu_clocks.sh" reset

# ---------------------------------------------------------------------------
# Aggregate.
# ---------------------------------------------------------------------------
echo "== Aggregation =="
if [ -s "$RESULTS_CSV" ]; then
    python3 "$SCRIPT_DIR/scripts/aggregate_results.py" "$RESULTS_CSV" --baseline seq
else
    echo "No results collected."
fi

echo
echo "Raw results: $RESULTS_CSV"
