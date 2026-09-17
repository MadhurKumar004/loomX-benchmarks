# AGENTS.md — loomX-benchmarks

## Scope

This is a standalone benchmark harness for `loomX`. It is intentionally separate from the ROSE/loomX source tree so that benchmark artifacts and large external suites do not pollute the compiler repo.

Only modify files inside this repository. Do not edit the parent `loomX/` tree from here.

## Setup

```bash
./setup.sh
```

Fetches PolyBench/C 4.2.1, Rodinia, and DataRaceBench into `suites/`.
The custom `interproc-microbench/` suite is already included.

## Running

```bash
./run_all.sh                       # everything
./run_suite.sh interproc-microbench
./run_suite.sh polybench
./run_suite.sh dataracebench
```

## Configuration

Edit `config.env` before running, especially:

- `LOOMX` — absolute path to the loomX binary.
- `LD_LIBRARY_PATH` — must include the directory containing `librose.so`.
- `COMPILER_CPU`, `COMPILER_GPU` — host and offload compilers.
- `GPU_ARCH` — e.g. `sm_80`. Leave empty to auto-detect.

All variables can also be exported on the command line.

## Output conventions

- Timing: `results/<suite>.csv`, aggregated with `scripts/aggregate_results.py --baseline seq`.
- Correctness: per-benchmark `results/<name>__{golden,out}.out` plus `scripts/check_correctness.py` output.
- DataRaceBench: `results/dataracebench.csv`.

Report full end-to-end wall-clock time, geometric mean across benchmarks, and failure counts. Do not report kernel-only time or arithmetic mean.

## Extending

To add a new suite:

1. Add fetching logic to `setup.sh` if the suite is external.
2. Add a case to `run_suite.sh` that defines `BENCHES`, `BENCH_SRC_DIR`, and any suite-specific compile flags.
3. Update `run_all.sh` and `README.md`.
