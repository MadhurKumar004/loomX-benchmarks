#!/usr/bin/env bash
# run_parboil.sh -- run a subset of Parboil sequential C base benchmarks
# through loomX and measure CPU OpenMP / GPU offload speedups.
#
# Usage:
#   ./run_parboil.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/config.env" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/config.env"
fi

LOOMX="${LOOMX:-$SCRIPT_DIR/../loomX/loomX}"
ROSE_INSTALL_PREFIX="${ROSE_INSTALL_PREFIX:-$SCRIPT_DIR/../rose-install}"
COMPILER_CPU="${COMPILER_CPU:-gcc}"
COMPILER_GPU="${COMPILER_GPU:-clang}"
GPU_ARCH="${GPU_ARCH:-sm_80}"
RUNS="${RUNS:-10}"
PARBOIL_DIR="${PARBOIL_DIR:-$SCRIPT_DIR/suites/parboil}"

if [ ! -d "$PARBOIL_DIR" ]; then
    echo "ERROR: $PARBOIL_DIR not found. Run ./setup.sh --refresh-parboil or set PARBOIL_DIR." >&2
    exit 1
fi

mkdir -p out/parboil_patched bin results

PATCH_DIR="$SCRIPT_DIR/patches/parboil"
WORK_DIR="$SCRIPT_DIR/out/parboil_patched"

# Patched parboil.h without OpenCL includes.
cp "$PATCH_DIR/parboil.h" "$WORK_DIR/parboil.h"

# Each entry: "bench_name|source_dir_relative_to_PARBOIL_DIR|source_files|run_args"
BENCHES=(
    "cp|benchmarks/cp/src/base|main.c cenergy.c|"
)

RESULTS_CSV="results/parboil.csv"
rm -f "$RESULTS_CSV"
echo "name,config,time" > "$RESULTS_CSV"

# Extract the last floating-point token from Parboil's timer output.
extract_time() {
    awk 'NF { for (i=NF; i>0; i--) if ($i ~ /^[0-9]+(\.[0-9]+)?$/) { print $i; exit } }'
}

run_one() {
    local name="$1"
    local src_dir_rel="$2"
    local src_files="$3"
    local bargs="$4"
    local src_dir="$PARBOIL_DIR/$src_dir_rel"
    local work_subdir="$WORK_DIR/$name"
    mkdir -p "$work_subdir"

    # Copy sources into the work directory and apply portability patches.
    local loomx_srcs=()
    for f in $src_files; do
        cp "$src_dir/$f" "$work_subdir/$f"
        # Stencil's file.c uses uint32_t which ROSE's system header setup
        # does not expose; fall back to unsigned int.
        if [ "$name" = "stencil" ] && [ "$f" = "file.c" ]; then
            sed -i 's/uint32_t/unsigned int/g' "$work_subdir/$f"
        fi
        # sad_cpu.c uses printf without including stdio.h.
        if [ "$name" = "sad" ] && [ "$f" = "sad_cpu.c" ]; then
            sed -i '1i#include <stdio.h>' "$work_subdir/$f"
        fi
        loomx_srcs+=("$work_subdir/$f")
    done

    local generated="$work_subdir/${name}_loomx.c"
    local generated_files=()

    echo "== $name =="

    # --- Sequential reference ---
    echo "  generating sequential..."
    for f in $src_files; do
        "$LOOMX" --cpu-only \
            -I"$ROSE_INSTALL_PREFIX/include/clang" \
            -I"$WORK_DIR" \
            -I"$src_dir" \
            "$work_subdir/$f" \
            -o "$work_subdir/${f%.c}_loomx.c" >/dev/null 2>&1 || {
            echo "  FAIL: loomX sequential generation failed for $f"
            return
        }
        generated_files+=("$work_subdir/${f%.c}_loomx.c")
    done
    if "$COMPILER_CPU" -O3 -I"$WORK_DIR" -I"$src_dir" \
            "${generated_files[@]}" \
            "$PARBOIL_DIR/common/src/args.c" "$PARBOIL_DIR/common/src/parboil.c" \
            -lm -o "bin/${name}__seq" >/dev/null 2>&1; then
        echo "  ok  $name/seq"
    else
        echo "  FAIL: $name/seq compilation failed"
        return
    fi

    # --- CPU OpenMP ---
    echo "  generating cpu_omp..."
    for f in $src_files; do
        "$LOOMX" \
            -I"$ROSE_INSTALL_PREFIX/include/clang" \
            -I"$WORK_DIR" \
            -I"$src_dir" \
            "$work_subdir/$f" \
            -o "$work_subdir/${f%.c}_loomx.c" >/dev/null 2>&1 || {
            echo "  FAIL: loomX cpu_omp generation failed for $f"
            return
        }
    done
    # Use the offload clang for CPU OpenMP so the binary links against the
    # LLVM libomp available in this environment.
    if "$COMPILER_GPU" -O3 -fopenmp -I"$WORK_DIR" -I"$src_dir" \
            "${generated_files[@]}" \
            "$PARBOIL_DIR/common/src/args.c" "$PARBOIL_DIR/common/src/parboil.c" \
            -lm -o "bin/${name}__cpu_omp" >/dev/null 2>&1; then
        echo "  ok  $name/cpu_omp"
    else
        echo "  FAIL: $name/cpu_omp compilation failed"
        return
    fi

    # --- GPU naive ---
    echo "  generating gpu_naive..."
    for f in $src_files; do
        "$LOOMX" --gpu-naive \
            -I"$ROSE_INSTALL_PREFIX/include/clang" \
            -I"$WORK_DIR" \
            -I"$src_dir" \
            "$work_subdir/$f" \
            -o "$work_subdir/${f%.c}_loomx.c" >/dev/null 2>&1 || {
            echo "  FAIL: loomX gpu_naive generation failed for $f"
            return
        }
        # ROSE's unparser can turn float constants like 0.0f into 0.00000L,
        # which NVPTX does not support. Convert them back to float suffixes.
        sed -i 's/\([0-9]*\.[0-9]*\)L/\1f/g' "$work_subdir/${f%.c}_loomx.c"
    done
    if "$COMPILER_GPU" -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
            -Xopenmp-target -march="$GPU_ARCH" \
            -I"$WORK_DIR" -I"$src_dir" \
            "${generated_files[@]}" \
            "$PARBOIL_DIR/common/src/args.c" "$PARBOIL_DIR/common/src/parboil.c" \
            -lm -o "bin/${name}__gpu_naive" >/dev/null 2>&1; then
        echo "  ok  $name/gpu_naive"
    else
        echo "  FAIL: $name/gpu_naive compilation failed"
    fi

    # --- GPU profitable ---
    echo "  generating gpu_profitable..."
    for f in $src_files; do
        "$LOOMX" --gpu-profitable \
            -I"$ROSE_INSTALL_PREFIX/include/clang" \
            -I"$WORK_DIR" \
            -I"$src_dir" \
            "$work_subdir/$f" \
            -o "$work_subdir/${f%.c}_loomx.c" >/dev/null 2>&1 || {
            echo "  FAIL: loomX gpu_profitable generation failed for $f"
            return
        }
        sed -i 's/\([0-9]*\.[0-9]*\)L/\1f/g' "$work_subdir/${f%.c}_loomx.c"
    done
    if "$COMPILER_GPU" -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
            -Xopenmp-target -march="$GPU_ARCH" \
            -I"$WORK_DIR" -I"$src_dir" \
            "${generated_files[@]}" \
            "$PARBOIL_DIR/common/src/args.c" "$PARBOIL_DIR/common/src/parboil.c" \
            -lm -o "bin/${name}__gpu_profitable" >/dev/null 2>&1; then
        echo "  ok  $name/gpu_profitable"
    else
        echo "  FAIL: $name/gpu_profitable compilation failed"
    fi

    # Run timing for each successfully built config.
    for cfg in seq cpu_omp gpu_naive gpu_profitable; do
        local bin="bin/${name}__${cfg}"
        [ -x "$bin" ] || continue
        local times=()
        for ((r=0; r<RUNS; r++)); do
            local t
            t=$(./"$bin" $bargs 2>/dev/null | extract_time) || t=""
            if [ -n "$t" ]; then
                times+=("$t")
            fi
        done
        if [ "${#times[@]}" -gt 0 ]; then
            local median
            median=$(printf '%s\n' "${times[@]}" | sort -n | awk 'NF{a[NR]=$1} END{print (NR%2?a[int(NR/2)+1]:(a[NR/2]+a[NR/2+1])/2)}')
            echo "$name,$cfg,$median" >> "$RESULTS_CSV"
            echo "  time $cfg = ${median}s"
        fi
    done
}

for spec in "${BENCHES[@]}"; do
    IFS='|' read -r name src_dir src_files bargs <<< "$spec"
    run_one "$name" "$src_dir" "$src_files" "$bargs"
done

echo "== Done. Results in $RESULTS_CSV =="
