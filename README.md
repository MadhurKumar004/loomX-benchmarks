# loomX Benchmarks

A standalone benchmark repository for the [loomX](https://github.com/rose-compiler/rose/tree/develop/tools/loomX) source-to-source parallelizer.

This repo contains:

- A custom **interproc-microbench** suite that stresses real function calls inside hot loops.
- Scripts to fetch three widely-used external benchmark suites:
  - **PolyBench/C 4.2.1** – regular affine kernels
  - **Rodinia** – realistic OpenMP/CUDA apps
  - **DataRaceBench** – LLNL data-race yes/no ground truth
- A unified harness that runs every suite through loomX, checks correctness, and aggregates timing results.

## Directory layout

```
loomX-benchmarks/
├── setup.sh                 # fetch external suites
├── run_all.sh               # run every suite
├── run_suite.sh             # run a single suite
├── config.env               # default environment variables
├── scripts/                 # helper Python / bash scripts
│   ├── check_correctness.py
│   ├── check_dataracebench.py
│   ├── bench_harness.py
│   ├── aggregate_results.py
│   └── lock_gpu_clocks.sh
└── suites/
    └── interproc-microbench/   # included custom tests
    # polybench, rodinia, dataracebench are created by setup.sh
```

## Quick start

### 1. Clone / configure

```bash
cd loomX-benchmarks
```

Edit `config.env` and set `LOOMX` to the path of your loomX binary:

```bash
export LOOMX=/path/to/loomX
```

Also adjust `LD_LIBRARY_PATH` if `librose.so` is not on the default search path.

### 2. Fetch external suites

```bash
./setup.sh
```

This populates `suites/polybench/`, `suites/rodinia/`, and `suites/dataracebench/`.

### 3. Run everything

```bash
./run_all.sh
```

Or run individual suites:

```bash
./run_suite.sh interproc-microbench
./run_suite.sh polybench
./run_suite.sh dataracebench
```

## What each suite does

### interproc-microbench

Generates four source variants per benchmark:

| config | meaning |
|--------|---------|
| `seq` | untouched sequential code, compiled `-O3` |
| `cpu_omp` | loomX `--cpu-only` output |
| `gpu_naive` | loomX `--gpu-naive` output (offload everything safe) |
| `gpu_profitable` | loomX `--gpu-profitable` output (profitability gate) |

Each variant is compiled, checked numerically against `seq`, and timed.

### polybench

Runs a representative subset of PolyBench/C 4.2.1 BLAS kernels (`gemm`, `syrk`, `syr2k`) through the same four-config pipeline.

### dataracebench

Runs loomX on every `DRB###-*-orig-{yes,no}.c` file and checks whether loomX correctly rejects `-yes` (racy) cases and accepts `-no` (race-free) cases. This is a correctness-only pass; no timing is performed.

## Overriding defaults

All environment variables in `config.env` can be overridden on the command line:

```bash
LOOMX=/opt/loomX/bin/loomX \
COMPILER_CPU=gcc \
COMPILER_GPU=clang \
GPU_ARCH=sm_80 \
RUNS=10 \
BENCH_ARGS=200000 \
LOCK_CLOCKS=yes \
./run_all.sh
```

## Results

Timing results are written to `results/<suite>.csv`. Use the aggregator to see speedups:

```bash
python3 scripts/aggregate_results.py results/interproc-microbench.csv --baseline seq
python3 scripts/aggregate_results.py results/polybench.csv --baseline seq
```

DataRaceBench results are written to `results/dataracebench.csv` with per-file expected/accept/reject labels.

## Notes

- GPU configs require a clang built with OpenMP offloading support and the matching `libomptarget-nvptx.bc` bitcode. If clang cannot target `nvptx64-nvidia-cuda`, the GPU variants are skipped and only CPU numbers are reported.
- `LOCK_CLOCKS=yes` locks NVIDIA GPU clocks for reproducible timing but requires root.
- Rodinia is fetched but not yet exercised by the default `run_all.sh` pipeline; you can extend `run_suite.sh` with a `rodinia` case if needed.
