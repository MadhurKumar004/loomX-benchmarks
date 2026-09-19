#!/usr/bin/env python3
"""
check_dataracebench.py -- evaluate loomX's parallel-safety judgments against
DataRaceBench's yes/no ground truth.

DataRaceBench files are named like:
    DRB###-name-orig-yes.c   -- contains a data race (loomX should reject)
    DRB###-name-orig-no.c    -- race-free (loomX should accept)

We run loomX on each file and count "accepted" (any OpenMP pragma inserted in
a loop body) vs "rejected".  The script reports false positives, false
negatives, and an overall accuracy score.

Usage:
    python3 check_dataracebench.py \
        --loomx /path/to/loomX \
        --suite /path/to/dataracebench/micro-benchmarks \
        --output drb_results.csv
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile


def count_openmp_pragmas(content):
    """Count OpenMP directives while ignoring comments and other pragmas."""
    return len(re.findall(r"^[ \t]*#[ \t]*pragma[ \t]+omp\b", content,
                          flags=re.MULTILINE))


def strip_parallelization_pragmas(content):
    """Blank execution annotations while retaining line numbers and semantics."""
    removable = re.compile(
        r"^[ \t]*#[ \t]*pragma[ \t]+omp[ \t]+"
        r"(?:parallel|for|target|teams|distribute|simd|task|sections)\b.*\n?",
        flags=re.MULTILINE,
    )
    def replace(match):
        text = match.group(0).lower()
        clause_text = re.sub(r"^[ \t]*#[ \t]*pragma[ \t]+", "", match.group(0), flags=re.IGNORECASE)
        if "nowait" in text or "ordered" in text:
            return "#pragma loomx semantic synchronization\n"
        return "#pragma loomx metadata " + clause_text.lstrip() + "\n"

    return removable.sub(replace, content)


def target_loop_lines(content):
    """Return loop lines controlled by removable OpenMP execution pragmas."""
    lines = content.splitlines()
    targets = []
    pragma = re.compile(
        r"^[ \t]*#[ \t]*pragma[ \t]+(?:loomx metadata[ \t]+)?"
        r"(?:omp[ \t]+)?(?:parallel|for|target|teams|distribute|simd|task|sections)\b"
    )
    for index, line in enumerate(lines):
        if not pragma.match(line):
            continue
        for candidate in range(index + 1, len(lines)):
            stripped = lines[candidate].strip()
            if not stripped or stripped.startswith("//"):
                continue
            if re.match(r"^(for|while|do)\b", stripped):
                targets.append(candidate + 1)
            break
    return targets


def run_loomx(loomx_path, src_path, mode="cpu-only", evaluation="preserve",
              strict_race_safety=False):
    """Run loomX and report whether it added an OpenMP directive.

    Existing OpenMP directives are semantic input. They must not be counted as
    loomX acceptance, because pragma-aware loomX intentionally preserves and
    skips those regions.

    Output is written to a temporary directory so stale .loomx.c files do not
    affect later runs or get scanned as DRB sources.
    """
    with tempfile.TemporaryDirectory() as td:
        analysis_src = src_path
        target_lines = []
        if evaluation == "stripped-analysis":
            analysis_src = os.path.join(td, os.path.basename(src_path))
            with open(src_path) as source_file:
                source = source_file.read()
            with open(analysis_src, "w") as analysis_file:
                analysis_file.write(strip_parallelization_pragmas(source))
            with open(analysis_src) as analysis_file:
                target_lines = target_loop_lines(analysis_file.read())

        out_path = os.path.join(td, os.path.basename(src_path) + ".loomx.c")
        command = [loomx_path, f"--{mode}", analysis_src, "-o", out_path]
        if evaluation == "stripped-analysis":
            command = [loomx_path, "--analyze-only", analysis_src]
        if strict_race_safety:
            command.insert(1, "--strict-race-safety")
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
        except Exception as e:
            print(f"  [warn] loomX failed on {src_path}: {e}", file=sys.stderr)
            return {
                "added": False,
                "preexisting": False,
                "input_pragmas": 0,
                "output_pragmas": 0,
                "failed": True,
            }

        if evaluation == "stripped-analysis":
            parallelized_lines = {
                int(match.group(1))
                for match in re.finditer(r"PARALLELIZED line (\d+)",
                                         completed.stdout.decode(errors="replace"))
            }
            return {
                "added": bool(target_lines) and all(
                    line in parallelized_lines for line in target_lines
                ),
                "preexisting": False,
                "input_pragmas": 0,
                "output_pragmas": len(parallelized_lines),
                "failed": completed.returncode != 0,
            }

        if not os.path.exists(out_path):
            return {
                "added": False,
                "preexisting": False,
                "input_pragmas": 0,
                "output_pragmas": 0,
                "failed": True,
            }

        with open(analysis_src) as source_file:
            source = source_file.read()
        with open(out_path) as output_file:
            content = output_file.read()

        input_count = count_openmp_pragmas(source)
        output_count = count_openmp_pragmas(content)
        return {
            "added": output_count > input_count,
            "preexisting": input_count > 0,
            "input_pragmas": input_count,
            "output_pragmas": output_count,
            "failed": False,
        }


def parse_label(filename):
    """Return ('yes'|'no'|None, base_name) from a DRB filename."""
    m = re.match(r"(DRB\d+-.+)-(orig|omp)-(yes|no)\.c$", filename)
    if not m:
        return None, None
    return m.group(3), m.group(1) + "-" + m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loomx", required=True, help="path to loomX translator")
    ap.add_argument("--suite", required=True, help="path to dataracebench micro-benchmarks dir")
    ap.add_argument("--mode", default="cpu-only", help="loomX mode: cpu-only, gpu-naive, gpu-profitable")
    ap.add_argument("--output", default="dataracebench_results.csv")
    ap.add_argument("--evaluation", choices=("preserve", "stripped-analysis"),
                    default="preserve",
                    help="preserve existing pragmas or strip only execution pragmas")
    ap.add_argument("--strict-race-safety", action="store_true",
                    help="reject reductions and implicit scalar privatization")
    args = ap.parse_args()

    results = []
    accepted = {"yes": 0, "no": 0}
    rejected = {"yes": 0, "no": 0}
    false_positives = []
    false_negatives = []
    preserved = []
    evaluated = []

    files = sorted(f for f in os.listdir(args.suite) if f.endswith(".c"))
    print(f"Scanning {len(files)} DRB sources in {args.suite} ...")

    for filename in files:
        label, base = parse_label(filename)
        if label is None:
            continue

        src = os.path.join(args.suite, filename)
        pragma_result = run_loomx(args.loomx, src, args.mode, args.evaluation,
                      args.strict_race_safety)
        parallelized = pragma_result["added"]

        if args.evaluation == "preserve" and pragma_result["preexisting"]:
            preserved.append(filename)
            results.append({
                "file": filename,
                "expected": "preexisting-pragma",
                "actual": "preserved",
                "correct": True,
            })
            continue

        if parallelized:
            accepted[label] += 1
            if label == "yes":
                false_positives.append(filename)
        else:
            rejected[label] += 1
            if label == "no":
                false_negatives.append(filename)

        results.append({
            "file": filename,
            "expected": "reject" if label == "yes" else "accept",
            "actual": "accept" if parallelized else "reject",
            "correct": (label == "yes" and not parallelized) or (label == "no" and parallelized),
        })
        evaluated.append(filename)

    total = len(evaluated)
    correct = sum(1 for r in results if r["file"] in evaluated and r["correct"])

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "expected", "actual", "correct"])
        w.writeheader()
        w.writerows(results)

        print(f"\nMode: {args.mode}  Evaluation: {args.evaluation}"
            f"  Strict race safety: {args.strict_race_safety}")
    print(f"Total evaluated: {total}")
    print(f"Correct:         {correct} ({100.0*correct/total:.1f}%)" if total else "N/A")
    print(f"False positives (accepted a -yes race): {len(false_positives)}")
    print(f"False negatives (rejected a -no safe):  {len(false_negatives)}")
    print(f"Preserved pre-existing pragma files:      {len(preserved)}")
    print(f"\nAccepted -no:  {accepted['no']}  Rejected -no:  {rejected['no']}")
    print(f"Accepted -yes: {accepted['yes']}  Rejected -yes: {rejected['yes']}")

    if false_positives:
        print("\nFalse positives:")
        for f in false_positives:
            print(f"  {f}")
    if false_negatives:
        print("\nFalse negatives:")
        for f in false_negatives[:20]:  # truncate long lists
            print(f"  {f}")
        if len(false_negatives) > 20:
            print(f"  ... and {len(false_negatives) - 20} more")

    print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
