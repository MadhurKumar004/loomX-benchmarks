# loomX Benchmarks

A standalone benchmark repository for the [loomX](https://github.com/rose-compiler/rose/tree/develop/tools/loomX) source-to-source parallelizer.

This repo contains:

- A custom **interproc-microbench** suite that stresses real function calls inside hot loops.
- Three widely-used external benchmark suites, vendored in `suites/`:
  - **PolyBench/C 4.2.1** – regular affine kernels
  - **Rodinia** – realistic OpenMP/CUDA apps
  - **DataRaceBench** – LLNL data-race yes/no ground truth
- A unified harness that runs every suite through loomX, checks correctness against the sequential output or the suite's own verification, reports failures, and aggregates timing results.

## Directory layout

```
loomX-benchmarks/
├── setup.sh                 # verify / refresh vendored suites
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
    ├── interproc-microbench/   # included custom tests
    ├── polybench/              # vendored PolyBench/C 4.2.1
    ├── rodinia/                # vendored Rodinia
    └── dataracebench/          # vendored DataRaceBench
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

### 2. Verify vendored suites

```bash
./setup.sh
```

This checks that `suites/polybench/`, `suites/rodinia/`, and `suites/dataracebench/` are present. The suites are already included in this repository, so no network access is required. To update them from upstream instead, run `./setup.sh --refresh`.

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

## Correctness caveats

The harness reports failures as well as passes; do not assume every parallel output matches the sequential golden output.

- **PolyBench / interproc-microbench**: numerical diff against `seq` with relaxed tolerance. Failing configs are excluded from speedup aggregates.
- **NPB**: uses the benchmark's own `Verification = SUCCESSFUL/UNSUCCESSFUL` line. Currently BT and EP fail verification on CPU OpenMP and GPU configs at the problem sizes we run; NPB's epsilon (~1e-8) rejects the small differences introduced by parallel reduction reordering.
- **Rodinia / LLVM Test Suite / Loop-Fission**: validated by successful execution and output shape, not by bitwise or tight numerical equality, because these apps print timing or reduction-order-sensitive results.
- **Parboil**: `cp` runs end-to-end and produces output; other base kernels need input-data handling before they can be validated.

## Notes

- GPU configs require a clang built with OpenMP offloading support and the matching `libomptarget-nvptx.bc` bitcode. If clang cannot target `nvptx64-nvidia-cuda`, the GPU variants are skipped and only CPU numbers are reported.
- `LOCK_CLOCKS=yes` locks NVIDIA GPU clocks for reproducible timing but requires root.
- Rodinia is included but not yet exercised by the default `run_all.sh` pipeline; you can extend `run_suite.sh` with a `rodinia` case if needed.
- Because the external suites are vendored, this repository is larger than a fetch-on-demand version. If size matters, delete `suites/rodinia/` or `suites/dataracebench/` and run `./setup.sh --refresh` to fetch only what you need.
