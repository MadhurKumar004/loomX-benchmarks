#!/usr/bin/env python3
"""
decision_regret.py -- score loomX's offload decision, not the parallelizer.

For a given benchmark this needs three timings of the *same* source:

  cpu_forced     forced-CPU arm   (loomX --cpu-forced)
  gpu_naive      forced-GPU arm   (loomX --gpu-naive)
  gpu_profitable loomX's own cost-model decision

With T_cpu, T_gpu and T_choice:

  T_best  = min(T_cpu, T_gpu)
  regret  = T_best / T_choice

Reading the result:

  regret > 1   loomX was slower than the better uniform arm. This is the case
                worth investigating: the cost model picked a losing uniform
                arm, or picked GPU where CPU was faster, and no amount of
                parallelisation quality can recover it.
  regret = 1   loomX matched the better uniform arm.
  regret < 1   loomX beat the better uniform arm.

  Caveat, and it matters: T_best is a min over the uniform arms, so a small
  T_best makes the ratio look flattering.  When loomX's choice is slower than
  *both* arms the ratio can still fall below 1 and a plain reading would call
  that a win.  Example: T_cpu=0.10, T_gpu=0.80, T_choice=0.90 gives
  regret=0.11 even though loomX lost to every alternative.  Such rows are
  therefore detected separately and always counted as losses, never wins.
  "Beat both arms" is only claimed when T_choice < min(T_cpu, T_gpu).

Because a uniform arm can in principle beat a mixed decision, values below 1
are reported as wins and values above 1 as losses, and the two are summarised
separately. A single arithmetic mean would let a large win cancel a large loss
and hide exactly the regressions worth chasing.

Note that --cpu-only is NOT a valid CPU arm here. It stays subject to the FLOP
threshold, so on a small loop it declines to thread at all and would hand loomX
credit for declining to parallelise. cpu_forced is the honest arm.

Usage:
  python3 decision_regret.py results.csv
  python3 decision_regret.py results.csv --correctness-csv results.correctness.csv
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

CPU_ARM = "cpu_forced"
GPU_ARM = "gpu_naive"
CHOICE_ARM = "gpu_profitable"


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def load_correctness(csv_path):
    """Return a set of (bench, config) tuples that failed correctness."""
    failed = set()
    if not csv_path or not os.path.exists(csv_path):
        return failed
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("result", "").strip().upper() == "FAIL":
                failed.add((row["name"].strip(), row["config"].strip()))
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="raw results CSV from bench_harness.py")
    ap.add_argument("--correctness-csv", default=None,
                    help="correctness CSV; failing configs are scored as pending")
    ap.add_argument("--threshold", type=float, default=1.05,
                    help="regret above this counts as a loss (default 1.05)")
    args = ap.parse_args()

    correctness_csv = args.correctness_csv
    if correctness_csv is None:
        base, _ = os.path.splitext(args.csv_path)
        correctness_csv = base + ".correctness.csv"
    failed = load_correctness(correctness_csv)

    data = defaultdict(dict)
    with open(args.csv_path) as f:
        for row in csv.DictReader(f):
            bench, _, config = row["label"].partition("__")
            try:
                data[bench][config] = float(row["median_s"])
            except (KeyError, TypeError, ValueError):
                continue

    scored = []
    pending = []
    for bench in sorted(data):
        cfgs = data[bench]
        missing = [c for c in (CPU_ARM, GPU_ARM, CHOICE_ARM) if c not in cfgs]
        if missing:
            pending.append((bench, "not run: " + ", ".join(missing)))
            continue
        # A numerically wrong arm cannot be compared on time: it is measuring a
        # different program, not a faster one.
        wrong = [c for c in (CPU_ARM, GPU_ARM, CHOICE_ARM) if (bench, c) in failed]
        if wrong:
            pending.append((bench, "failed correctness: " + ", ".join(wrong)))
            continue
        t_cpu = cfgs[CPU_ARM]
        t_gpu = cfgs[GPU_ARM]
        t_choice = cfgs[CHOICE_ARM]
        t_best = min(t_cpu, t_gpu)
        if t_choice <= 0:
            pending.append((bench, "non-positive choice timing"))
            continue
        regret = t_best / t_choice
        # Guard against the min-in-the-denominator flattering a choice that
        # actually lost to every arm.
        worse_than_all = t_choice > t_cpu and t_choice > t_gpu
        scored.append((bench, t_cpu, t_gpu, t_choice, regret,
                       "cpu" if t_cpu < t_gpu else "gpu", worse_than_all))

    print("== Offload decision regret ==")
    print(f"{'benchmark':<40} {'T_cpu':>9} {'T_gpu':>9} {'T_loomx':>9} "
          f"{'regret':>7}  best")
    print("-" * 84)
    for bench, t_cpu, t_gpu, t_choice, regret, best, _w in scored:
        print(f"{bench:<40} {t_cpu:>9.4f} {t_gpu:>9.4f} {t_choice:>9.4f} "
              f"{regret:>7.3f}  {best}")

    worse = [s for s in scored if s[6]]
    wins = [s for s in scored
            if not s[6] and s[4] < 1.0 / args.threshold]
    losses = [s for s in scored if s not in wins and s not in worse]
    neutral = len(scored) - len(losses) - len(wins) - len(worse)

    print()
    print(f"scored:  {len(scored)}  (geomean regret {geomean([s[4] for s in scored]):.3f})")
    print(f"losses:  {len(losses)}  (regret > {args.threshold})")
    print(f"wins:    {len(wins)}  (regret < {1.0 / args.threshold:.3f})")
    print(f"neutral: {neutral}")
    if worse:
        print(f"worse-than-every-arm: {len(worse)}  (regret understates these; "
              f"counted as losses, not wins)")
        for s in worse:
            print(f"    {s[0]}: T_cpu={s[1]:.4f} T_gpu={s[2]:.4f} "
                  f"T_loomx={s[3]:.4f}")

    # Which side of the decision is responsible for the losses.  A loss where
    # CPU was the better arm but loomX went to GPU is a cost-model false
    # positive and is far more damaging than the reverse.  worse-than-every-arm
    # rows count here too: those are the most severe regressions of all.
    regressed = losses + worse
    fp = [s for s in regressed if s[5] == "cpu"]
    fn = [s for s in regressed if s[5] == "gpu"]
    if fp or fn:
        print()
        print(f"  false positives (CPU was faster, loomX offloaded): {len(fp)}")
        for s in fp:
            print(f"    {s[0]}")
        print(f"  false negatives (GPU was faster, loomX stayed on CPU): {len(fn)}")
        for s in fn:
            print(f"    {s[0]}")

    if pending:
        print()
        print("pending (unscoreable, arm missing or numerically wrong):")
        for bench, why in pending:
            print(f"  {bench:<40} {why}")

    if not scored:
        print()
        print("No benchmark had all three arms. Run the suite with the full "
              "config matrix (see CONFIGS in run_suite.sh).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
