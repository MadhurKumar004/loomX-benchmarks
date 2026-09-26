# AGENTS.md — loomX-benchmarks

## Scope

This is a standalone benchmark harness for `loomX`. It is intentionally separate from the ROSE/loomX source tree so that benchmark artifacts and large external suites do not pollute the compiler repo.

Only modify files inside this repository. Do not edit the parent `loomX/` tree from here.

## Setup

```bash
./setup.sh
```

Verifies that PolyBench/C 4.2.1, Rodinia, and DataRaceBench are present under `suites/`.
All suites are vendored in this repository; no network fetch is needed by default.
Run `./setup.sh --refresh` to re-clone them from upstream (destructive).

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

### Configurations and decision regret

`run_suite.sh` defines its arms in two arrays near the top, `CONFIGS` and
`CHECK_CONFIGS`, and everything else derives from them: do not re-list a config
name inline. The arms are

- `seq` — untouched baseline, the denominator for every speedup.
- `cpu_omp` — the shipped default, which still honours the FLOP thresholds.
- `cpu_forced` — CPU OpenMP for every safe loop. The forced-CPU arm.
- `gpu_naive` — offload every safe loop, ignoring profitability.
- `gpu_profitable` — the cost model chooses.

`cpu_forced` and `gpu_naive` are ablations, not recommendations. They exist so
the choice made by `gpu_profitable` can be scored against both forced arms:

```bash
./run_suite.sh interproc-microbench
python3 scripts/decision_regret.py results/interproc-microbench.csv \
    --correctness-csv results/interproc-microbench.correctness.csv
```

`decision_regret.py` reports `min(T_cpu, T_gpu) / T_choice` per benchmark, where
the choice is the `cpu_forced`/`gpu_naive` arm the model was free to pick. A
correctness failure downgrades a row to `pending` rather than letting a fast but
wrong kernel count as a win, and a choice slower than *both* forced arms is
reported as a loss instead of being scored as a large speedup. Report the
geometric mean across benchmarks, not the arithmetic mean.

`interproc-microbench` is the suite that exercises pointer arguments. Every
kernel there allocates with `malloc` and walks it through a `double *`, so it is
the regression net for offload data-mapping shape: a pointer mapped as a bare
name transfers the pointer value instead of the array, and the device then
dereferences whatever it finds. All 12 of its kernels must PASS correctness on
`gpu_naive` and `gpu_profitable` before a timing number from that suite means
anything.
- Correctness: per-benchmark `results/<name>__{golden,out}.out` plus `scripts/check_correctness.py` output.
- DataRaceBench: `results/dataracebench.csv`.

Report full end-to-end wall-clock time, geometric mean across benchmarks, and failure counts. Do not report kernel-only time or arithmetic mean.

## Extending

To add a new suite:

1. Add fetching logic to `setup.sh` if the suite is external.
2. Add a case to `run_suite.sh` that defines `BENCHES`, `BENCH_SRC_DIR`, and any suite-specific compile flags.
3. Update `run_all.sh` and `README.md`.
