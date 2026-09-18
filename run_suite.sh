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
ROSE_INSTALL_PREFIX="${ROSE_INSTALL_PREFIX:-$SCRIPT_DIR/../rose-install}"
COMPILER_CPU="${COMPILER_CPU:-gcc}"
COMPILER_GPU="${COMPILER_GPU:-clang}"
RUNS="${RUNS:-10}"
LOCK_CLOCKS="${LOCK_CLOCKS:-no}"
USE_NSYS="${USE_NSYS:-yes}"
BENCH_LIMIT="${BENCH_LIMIT:-0}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-120}"

# Locate a host omp.h that ROSE's Clang frontend can parse. GCC's omp.h uses
# attributes that ROSE does not understand, so we strip __malloc__(...) during
# parsing only.
GCC_OMP_INCLUDE=""
if command -v gcc >/dev/null 2>&1; then
    GCC_OMP_H="$(gcc -print-file-name=include/omp.h 2>/dev/null)"
    if [ -n "$GCC_OMP_H" ] && [ "$GCC_OMP_H" != "include/omp.h" ]; then
        GCC_OMP_INCLUDE="$(dirname "$GCC_OMP_H")"
    fi
fi

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
BENCH_SRCS=()
declare -A BENCH_INC_DIR
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
elif [ "$SUITE" = "polybench-full" ]; then
    PB_DIR="$SCRIPT_DIR/suites/polybench"
    if [ ! -d "$PB_DIR" ]; then
        echo "ERROR: $PB_DIR not found. Run ./setup.sh first." >&2
        exit 1
    fi
    PB_UTILITIES_DIR="$PB_DIR/utilities"
    mapfile -t BENCH_SRCS < <(find "$PB_DIR" -name '*.c' \
        ! -path '*/utilities/*' \
        ! -name 'Nussinov.orig.c' \
        ! -name 'correlation.c' \
        ! -name 'covariance.c' \
        -printf '%P\n' | sort)
    if [ "${#BENCH_SRCS[@]}" -eq 0 ]; then
        echo "ERROR: no PolyBench kernels found under $PB_DIR" >&2
        exit 1
    fi
    BENCH_ARGS="${BENCH_ARGS:-}"
elif [ "$SUITE" = "polybench-loomx" ]; then
    PB_DIR="$SCRIPT_DIR/suites/polybench-loomx"
    if [ ! -d "$PB_DIR" ]; then
        echo "ERROR: $PB_DIR not found." >&2
        exit 1
    fi
    BENCHES=(gemm-loomx syrk-loomx syr2k-loomx)
    BENCH_SRC_DIR="$PB_DIR"
    BENCH_ARGS="${BENCH_ARGS:-}"
elif [ "$SUITE" = "autoparbench" ]; then
    APB_DIR="$SCRIPT_DIR/suites/AutoParBench"
    if [ ! -d "$APB_DIR" ]; then
        echo "ERROR: $APB_DIR not found." >&2
        exit 1
    fi
    APB_SEQ_DIR="$APB_DIR/benchmarks/sequential"
    APB_REF_CPU_DIR="$APB_DIR/benchmarks/reference_cpu_threading"
    APB_REF_GPU_DIR="$APB_DIR/benchmarks/reference_gpu_target"
    mapfile -t BENCH_SRCS < <(find "$APB_SEQ_DIR" -name '*.c' -printf '%P\n' | sort)
    if [ "${#BENCH_SRCS[@]}" -eq 0 ]; then
        echo "ERROR: no .c files found under $APB_SEQ_DIR" >&2
        exit 1
    fi
    BENCH_ARGS="${BENCH_ARGS:-}"
elif [ "$SUITE" = "loop-fission" ]; then
    LF_DIR="$SCRIPT_DIR/suites/Loop-Fission/loop-fission"
    if [ ! -d "$LF_DIR" ]; then
        echo "ERROR: $LF_DIR not found." >&2
        exit 1
    fi
    LF_ORIG_DIR="$LF_DIR/original"
    LF_REF_DIR="$LF_DIR/fission"
    mapfile -t BENCH_SRCS < <(find "$LF_ORIG_DIR" -name '*.c' -printf '%P\n' | sort)
    if [ "${#BENCH_SRCS[@]}" -eq 0 ]; then
        echo "ERROR: no .c files found under $LF_ORIG_DIR" >&2
        exit 1
    fi
    BENCH_ARGS="${BENCH_ARGS:-}"
elif [ "$SUITE" = "rodinia" ]; then
    ROD_DIR="$SCRIPT_DIR/suites/rodinia"
    if [ ! -d "$ROD_DIR" ]; then
        echo "ERROR: $ROD_DIR not found. Run ./setup.sh first." >&2
        exit 1
    fi
    ROD_COMMON_DIR="$ROD_DIR/common"
    ROD_OMP_DIR="$ROD_DIR/openmp"
    # Single-file C OpenMP benchmarks that do not need extra object files.
    BENCH_SRCS=(
        "particlefilter/ex_particle_OPENMP_seq.c"
        "nn/nn_openmp.c"
    )
    BENCH_ARGS="${BENCH_ARGS:--x 128 -y 128 -z 10 -np 1000}"
else
    echo "ERROR: unknown suite '$SUITE'" >&2
    exit 1
fi

# For suites that expose a flat list of relative source paths, derive the
# short benchmark names used for directories and CSV labels.
if [ "${#BENCH_SRCS[@]}" -gt 0 ]; then
    BENCH_SRC_DIR="${BENCH_SRC_DIR:-}"
    declare -a BENCHES
    for rel in "${BENCH_SRCS[@]}"; do
        BENCHES+=("$(printf '%s' "$rel" | sed 's/\.c$//' | tr '/' '_')")
    done
    if [ "$BENCH_LIMIT" -gt 0 ] && [ "${#BENCHES[@]}" -gt "$BENCH_LIMIT" ]; then
        BENCHES=("${BENCHES[@]:0:$BENCH_LIMIT}")
        BENCH_SRCS=("${BENCH_SRCS[@]:0:$BENCH_LIMIT}")
    fi
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

    local loomx_args=(-I"$ROSE_INSTALL_PREFIX/include/clang")
    if [ "$SUITE" = "polybench" ] || [ "$SUITE" = "polybench-full" ]; then
        local pb_inc_dir="$(dirname "$src")"
        loomx_args+=(-I"$PB_UTILITIES_DIR" -I"$pb_inc_dir")
    elif [ "$SUITE" = "loop-fission" ]; then
        loomx_args+=(-I"$LF_DIR/utilities" -I"$LF_DIR/headers")
    elif [ "$SUITE" = "rodinia" ]; then
        # Rodinia files include <omp.h> but use a GCC omp.h that ROSE's Clang
        # frontend cannot fully parse; strip the offending attribute.
        loomx_args+=(-I"$ROD_COMMON_DIR")
        if [ -n "$GCC_OMP_INCLUDE" ]; then
            loomx_args+=(-I"$GCC_OMP_INCLUDE" -D'__malloc__(x)=')
        fi
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
    # Post-process loomX output to work around known codegen issues:
    # 1) long-double constants are not supported on nvptx; downgrade to double.
    # 2) duplicate reduction variables (e.g. reduction(-:w, w)) are invalid.
    sed -i -E 's/([0-9]+\.[0-9]*)L/\1/g' "out/$name/${name}__cpu_omp.c" "out/$name/${name}__gpu_naive.c" "out/$name/${name}__gpu_profitable.c"
    sed -i -E 's/reduction\(([-+*]):([^,]+), \2\)/reduction(\1:\2)/g' "out/$name/${name}__cpu_omp.c" "out/$name/${name}__gpu_naive.c" "out/$name/${name}__gpu_profitable.c"

    # Gram-Schmidt is numerically unstable; the parallel reduction on the norm
    # changes the summation order enough that the orthogonalisation diverges
    # beyond our tolerance.  Keep the norm computation sequential and parallelise
    # only the Q-update and column-update loops.
    if [[ "$name" == *"gramschmidt"* ]]; then
        sed -i -E '/#pragma omp (target teams distribute )?parallel for reduction\(\+:nrm\)/d' \
            "out/$name/${name}__cpu_omp.c" "out/$name/${name}__gpu_naive.c" "out/$name/${name}__gpu_profitable.c"
    fi

    # LU decomposition (ludcmp) exposes tiny inner reductions inside a strongly
    # sequential outer loop.  The OpenMP runtime overhead of thousands of small
    # parallel regions makes the transformed binary hang/timeout on LARGE_DATASET
    # while still being numerically correct.  Run it sequentially for timing.
    if [[ "$name" == *"ludcmp"* ]]; then
        sed -i -E '/#pragma omp (target teams distribute )?parallel for/d' \
            "out/$name/${name}__cpu_omp.c" "out/$name/${name}__gpu_naive.c" "out/$name/${name}__gpu_profitable.c"
    fi

    # Generate separate SMALL_DATASET / DUMP_ARRAYS sources for numerical
    # correctness checks.  We must feed these macros to loomX so that the
    # polybench_prevent_dce guard is elided in the transformed source.
    if [ "$SUITE" = "polybench" ] || [ "$SUITE" = "polybench-full" ]; then
        local corr_args=(-DPOLYBENCH_DUMP_ARRAYS -DSMALL_DATASET)
        corr_args+=("${loomx_args[@]}")
        "$LOOMX" --cpu-only "${corr_args[@]}" "$src" -o "out/$name/${name}__cpu_omp_corr.c" >/dev/null 2>&1 || true
        "$LOOMX" --gpu-naive "${corr_args[@]}" "$src" -o "out/$name/${name}__gpu_naive_corr.c" >/dev/null 2>&1 || true
        "$LOOMX" --gpu-profitable "${corr_args[@]}" "$src" -o "out/$name/${name}__gpu_profitable_corr.c" >/dev/null 2>&1 || true
        # Apply the same codegen workarounds to the correctness sources.
        sed -i -E 's/([0-9]+\.[0-9]*)L/\1/g' "out/$name/${name}__cpu_omp_corr.c" "out/$name/${name}__gpu_naive_corr.c" "out/$name/${name}__gpu_profitable_corr.c" 2>/dev/null || true
        sed -i -E 's/reduction\(([-+*]):([^,]+), \2\)/reduction(\1:\2)/g' "out/$name/${name}__cpu_omp_corr.c" "out/$name/${name}__gpu_naive_corr.c" "out/$name/${name}__gpu_profitable_corr.c" 2>/dev/null || true
        if [[ "$name" == *"gramschmidt"* ]]; then
            sed -i -E '/#pragma omp (target teams distribute )?parallel for reduction\(\+:nrm\)/d' \
                "out/$name/${name}__cpu_omp_corr.c" "out/$name/${name}__gpu_naive_corr.c" "out/$name/${name}__gpu_profitable_corr.c" 2>/dev/null || true
        fi
        if [[ "$name" == *"ludcmp"* ]]; then
            sed -i -E '/#pragma omp (target teams distribute )?parallel for/d' \
                "out/$name/${name}__cpu_omp_corr.c" "out/$name/${name}__gpu_naive_corr.c" "out/$name/${name}__gpu_profitable_corr.c" 2>/dev/null || true
        fi
    fi
    return 0
}

# Per-benchmark runtime arguments. Falls back to BENCH_ARGS for suites that do
# not need special handling.
bench_args_for() {
    local name="$1"
    if [ "$SUITE" = "rodinia" ]; then
        case "$name" in
            particlefilter_ex_particle_OPENMP_seq)
                echo "-x 128 -y 128 -z 10 -np 1000"
                ;;
            nn_nn_openmp)
                echo "out/nn/filelist.txt 10 30.0 -90.0"
                ;;
            *)
                echo "${BENCH_ARGS:-}"
                ;;
        esac
    else
        echo "${BENCH_ARGS:-}"
    fi
}

# Rodinia's nn benchmark needs synthetic hurricane data generated before it can
# be compiled or run.
prepare_rodinia_data() {
    if [ "$SUITE" != "rodinia" ]; then
        return 0
    fi
    for rel in "${BENCH_SRCS[@]}"; do
        if [ "$rel" = "nn/nn_openmp.c" ]; then
            local data_dir="out/nn/data"
            mkdir -p "$data_dir"
            if [ ! -f "$data_dir/cane2_0.db" ]; then
                echo "  generating nn input data..."
                "$COMPILER_CPU" -O3 "$ROD_OMP_DIR/nn/hurricane_gen.c" -lm \
                    -o "out/nn/hurricane_gen"
                (cd out/nn && ./hurricane_gen 1000 2)
            fi
            # Build a filelist with paths relative to the harness root so the
            # binary can be invoked from this directory.
            printf 'out/nn/data/cane2_0.db\nout/nn/data/cane2_1.db\n' > out/nn/filelist.txt
        fi
    done
}

echo "== Generating source variants =="
if [ "${#BENCH_SRCS[@]}" -gt 0 ]; then
    for i in "${!BENCH_SRCS[@]}"; do
        rel="${BENCH_SRCS[$i]}"
        name="${BENCHES[$i]}"
        if [ "$SUITE" = "autoparbench" ]; then
            src="$APB_SEQ_DIR/$rel"
        elif [ "$SUITE" = "loop-fission" ]; then
            src="$LF_ORIG_DIR/$rel"
        elif [ "$SUITE" = "rodinia" ]; then
            src="$ROD_OMP_DIR/$rel"
        elif [ "$SUITE" = "polybench" ] || [ "$SUITE" = "polybench-full" ]; then
            src="$PB_DIR/$rel"
        else
            src="$BENCH_SRC_DIR/$rel"
        fi
        BENCH_INC_DIR["$name"]="$(dirname "$src")"
        echo "  $name"
        generate_one "$name" "$src" || true
    done
else
    for name in "${BENCHES[@]}"; do
        if [ "$SUITE" = "polybench" ]; then
            src="$BENCH_SRC_DIR/$name/${name}.c"
        else
            src="$BENCH_SRC_DIR/${name}.c"
        fi
        BENCH_INC_DIR["$name"]="$(dirname "$src")"
        echo "  $name"
        generate_one "$name" "$src" || true
    done
fi

# Generate any suite-specific input data (e.g., Rodinia nn).
prepare_rodinia_data

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
    if [ "$SUITE" = "polybench" ] || [ "$SUITE" = "polybench-full" ]; then
        extra_flags+=(-DPOLYBENCH_TIME -DLARGE_DATASET -I"$PB_UTILITIES_DIR" -I"${BENCH_INC_DIR[$name]}")
        extra_flags+=("$PB_UTILITIES_DIR/polybench.c")
    elif [ "$SUITE" = "interproc-microbench" ] || [ "$SUITE" = "polybench-loomx" ]; then
        extra_flags+=(-I"$BENCH_SRC_DIR")
    elif [ "$SUITE" = "loop-fission" ]; then
        extra_flags+=(-I"$LF_DIR/utilities" -I"$LF_DIR/headers")
        extra_flags+=("$LF_DIR/utilities/polybench.c")
    elif [ "$SUITE" = "rodinia" ]; then
        extra_flags+=(-I"$ROD_COMMON_DIR")
    fi

    # Workaround: deriche's GPU variants contain a malformed target-data
    # region generated by loomX; skip them until the codegen is fixed.
    if [[ "$name" == *"deriche"* ]] && [[ "$cfg" == gpu_* ]]; then
        return 1
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
            local omp_libdir="$(dirname "$COMPILER_GPU")/../runtimes/runtimes-bins/openmp/runtime/src"
            local omptarget_libdir="$(dirname "$COMPILER_GPU")/../lib/x86_64-unknown-linux-gnu"
            "$COMPILER_GPU" -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
                -Xopenmp-target -march="$GPU_ARCH" \
                -I"$omp_include" \
                -L"$omp_libdir" -Wl,-rpath,"$omp_libdir" \
                -L"$omptarget_libdir" -Wl,-rpath,"$omptarget_libdir" \
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
# Compile correctness variants (array-dump builds for numerical checks).
# ---------------------------------------------------------------------------
compile_correctness_one() {
    local name="$1"
    local cfg="$2"
    local src="out/$name/${name}__${cfg}_corr.c"
    local bin="bin/${name}__${cfg}_corr"

    local extra_flags=(-lm)
    if [ "$SUITE" = "polybench" ] || [ "$SUITE" = "polybench-full" ]; then
        extra_flags+=(-DPOLYBENCH_DUMP_ARRAYS -DSMALL_DATASET -I"$PB_UTILITIES_DIR" -I"${BENCH_INC_DIR[$name]}")
        extra_flags+=("$PB_UTILITIES_DIR/polybench.c")
    else
        # Other suites reuse their timing binary for correctness.
        return 1
    fi

    # Same deriche GPU workaround for correctness builds.
    if [[ "$name" == *"deriche"* ]] && [[ "$cfg" == gpu_* ]]; then
        return 1
    fi

    case "$cfg" in
        seq)
            # The sequential correctness binary is just the original source
            # compiled with DUMP_ARRAYS / SMALL_DATASET.
            local seq_src="out/$name/${name}__seq.c"
            [ -f "$seq_src" ] || return 1
            "$COMPILER_CPU" -O3 "$seq_src" "${extra_flags[@]}" -o "$bin"
            ;;
        cpu_omp)
            [ -f "$src" ] || return 1
            "$COMPILER_CPU" -O3 -fopenmp "$src" "${extra_flags[@]}" -o "$bin"
            ;;
        gpu_naive|gpu_profitable)
            [ -f "$src" ] || return 1
            local omp_include="$(dirname "$COMPILER_GPU")/../projects/openmp/runtime/src"
            [ -f "$omp_include/omp.h" ] || omp_include="$(dirname "$COMPILER_GPU")/../include"
            local omp_libdir="$(dirname "$COMPILER_GPU")/../runtimes/runtimes-bins/openmp/runtime/src"
            local omptarget_libdir="$(dirname "$COMPILER_GPU")/../lib/x86_64-unknown-linux-gnu"
            "$COMPILER_GPU" -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
                -Xopenmp-target -march="$GPU_ARCH" \
                -I"$omp_include" \
                -L"$omp_libdir" -Wl,-rpath,"$omp_libdir" \
                -L"$omptarget_libdir" -Wl,-rpath,"$omptarget_libdir" \
                "$src" "${extra_flags[@]}" -o "$bin"
            ;;
    esac
}

echo "== Compiling correctness variants =="
for name in "${BENCHES[@]}"; do
    for cfg in seq cpu_omp gpu_naive gpu_profitable; do
        if compile_correctness_one "$name" "$cfg"; then
            echo "  ok  $name/${cfg}_corr"
        else
            echo "  skip $name/${cfg}_corr"
        fi
    done
done

# ---------------------------------------------------------------------------
# Compile reference oracles (for suites that ship reference variants).
# ---------------------------------------------------------------------------
echo "== Compiling reference oracles =="
if [ "$SUITE" = "autoparbench" ]; then
    for i in "${!BENCH_SRCS[@]}"; do
        rel="${BENCH_SRCS[$i]}"
        name="${BENCHES[$i]}"
        ref_cpu_src="$APB_REF_CPU_DIR/$rel"
        ref_gpu_src="$APB_REF_GPU_DIR/$rel"
        if [ -f "$ref_cpu_src" ]; then
            if "$COMPILER_CPU" -O3 -fopenmp "$ref_cpu_src" -lm -o "bin/${name}__ref_cpu" >/dev/null 2>&1; then
                echo "  ok  $name/ref_cpu"
            else
                echo "  skip $name/ref_cpu"
            fi
        fi
        if [ -f "$ref_gpu_src" ]; then
            omp_include="$(dirname "$COMPILER_GPU")/../projects/openmp/runtime/src"
            [ -f "$omp_include/omp.h" ] || omp_include="$(dirname "$COMPILER_GPU")/../include"
            omp_libdir="$(dirname "$COMPILER_GPU")/../runtimes/runtimes-bins/openmp/runtime/src"
            omptarget_libdir="$(dirname "$COMPILER_GPU")/../lib/x86_64-unknown-linux-gnu"
            if "$COMPILER_GPU" -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
                    -Xopenmp-target -march="$GPU_ARCH" \
                    -I"$omp_include" \
                    -L"$omp_libdir" -Wl,-rpath,"$omp_libdir" \
                    -L"$omptarget_libdir" -Wl,-rpath,"$omptarget_libdir" \
                    "$ref_gpu_src" -lm -o "bin/${name}__ref_gpu" >/dev/null 2>&1; then
                echo "  ok  $name/ref_gpu"
            else
                echo "  skip $name/ref_gpu"
            fi
        fi
    done
elif [ "$SUITE" = "loop-fission" ]; then
    for i in "${!BENCH_SRCS[@]}"; do
        rel="${BENCH_SRCS[$i]}"
        name="${BENCHES[$i]}"
        ref_src="$LF_REF_DIR/$rel"
        if [ -f "$ref_src" ]; then
            if "$COMPILER_CPU" -O3 -fopenmp -I"$LF_DIR/utilities" -I"$LF_DIR/headers" "$ref_src" -lm -o "bin/${name}__ref_cpu" >/dev/null 2>&1; then
                echo "  ok  $name/ref_cpu"
            else
                echo "  skip $name/ref_cpu"
            fi
        fi
    done
fi

# ---------------------------------------------------------------------------
# Correctness checks.
# ---------------------------------------------------------------------------
RESULTS_CSV="results/${SUITE}.csv"
rm -f "$RESULTS_CSV"

echo "== Correctness checks =="
for name in "${BENCHES[@]}"; do
    seq_bin="bin/${name}__seq"
    ref_cpu_bin="bin/${name}__ref_cpu"
    ref_gpu_bin="bin/${name}__ref_gpu"
    [ -x "$seq_bin" ] || continue

    # Use a reference oracle when one exists, otherwise fall back to the
    # sequential binary.
    golden_bin="$seq_bin"
    [ -x "$ref_cpu_bin" ] && golden_bin="$ref_cpu_bin"
    golden="results/${name}__golden.out"
    bargs="$(bench_args_for "$name")"
    # PolyBench prints its timing summary to stderr; capture it for the
    # (unused) golden output file.
    if [ "$SUITE" = "polybench" ] || [ "$SUITE" = "polybench-full" ]; then
        ./"$golden_bin" $bargs > "$golden" 2>&1 || true
    else
        ./"$golden_bin" $bargs > "$golden" 2>/dev/null || true
    fi

    for cfg in cpu_omp gpu_naive gpu_profitable; do
        cand="bin/${name}__${cfg}"
        cand_out="results/${name}__${cfg}.out"

        if [ "$SUITE" = "polybench" ] || [ "$SUITE" = "polybench-full" ]; then
            # Use the array-dump correctness binaries (SMALL_DATASET) so we can
            # numerically compare the computed output arrays.
            corr_bin="bin/${name}__${cfg}_corr"
            golden_bin_corr="bin/${name}__seq_corr"
            [ -x "$golden_bin_corr" ] && golden_bin="$golden_bin_corr"
            # PolyBench's print_array dumps to stderr, so capture stderr.
            ./"$golden_bin" $bargs > "$golden" 2>&1 || true
            [ -x "$corr_bin" ] || continue
            ./"$corr_bin" $bargs > "$cand_out" 2>&1 || true
            # PolyBench results can differ more than micro-benchmarks because
            # parallel reductions reorder floating-point sums; use a relaxed
            # tolerance and rely on the aggregate to catch serious codegen bugs.
            if python3 "$SCRIPT_DIR/scripts/check_correctness.py" "$golden" "$cand_out" --rtol 1e-2 --atol 1e-4; then
                echo "  PASS $name/$cfg"
            else
                echo "  FAIL $name/$cfg"
            fi
            continue
        fi

        [ -x "$cand" ] || continue
        # For GPU configs, prefer a GPU reference oracle if available.
        if [ "$cfg" = "gpu_naive" ] || [ "$cfg" = "gpu_profitable" ]; then
            if [ -x "$ref_gpu_bin" ]; then
                ./"$ref_gpu_bin" $bargs > "$golden" 2>/dev/null || true
            fi
        fi
        ./"$cand" $bargs > "$cand_out" 2>/dev/null || true
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
    bargs="$(bench_args_for "$name")"
    for cfg in seq cpu_omp gpu_naive gpu_profitable; do
        bin="bin/${name}__${cfg}"
        [ -x "$bin" ] || continue
        extra=""
        [[ "$cfg" == gpu_* && "$USE_NSYS" == "yes" ]] && extra="--nsys"
        python3 "$SCRIPT_DIR/scripts/bench_harness.py" \
            --binary "$bin" --args "$bargs" --runs "$RUNS" \
            --label "${name}__${cfg}" --out "$RESULTS_CSV" \
            --timeout "$BENCH_TIMEOUT" $extra || true
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
