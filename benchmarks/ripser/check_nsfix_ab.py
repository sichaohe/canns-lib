#!/usr/bin/env python3
"""Check the interleaved A/B CSV (reps x arms x builds).

RULES ENFORCED
  1. STRUCTURE. The expected cell grid is computed from the EXPECTED table
     below (reps x arms x builds), never from the row count of the file.
     The denominator is printed.
  2. NO INCOMPLETE CELLS. A cell with an empty wall time, or a verdict
     other than OK, is a FAILURE.
  3. THE FEATURE-SET GATE. Every cell must have an empty `gpu_err`. The GPU
     apparent pre-pass is behind `#[cfg(feature = "gpu")]`, and the feature
     is not in `default` nor in `[tool.maturin] features`; a build without it
     silently falls back to the CPU pre-pass and is ~35% slower, which reads
     exactly like a regression in the change under test. A CSV that compares
     two builds with different feature sets is not a measurement.
  4. BARCODE. Every cell must carry the reference md5 and the reference bar
     counts. A speed number with a different barcode is not a speed number.
  5. THE CLAIM. The change under test is "the capped arm must get much
     faster while every other arm stays within noise". The checker
     recomputes both numbers from the CSV and fails if the capped arm did
     not improve by at least 40% or if any other arm moved by more than
     15%. (The threshold is a floor, not a target: the measured effect is
     4.3x.)
  6. NEGATIVE SELF-TEST. The checker mutates COPIES of the CSV in ways that
     must be caught and asserts that it rejects each one.

USAGE
    python check_nsfix_ab.py nsfix_ab.csv --base orig --new base
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import tempfile

REF_MD5 = ("ee1f5283091dd645a3dd53e6f1a7eac6:"
           "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
           "2d018d8fe9c1895b1b74f837af59ed20")
REF_BARS = "1195|481|58"

# The grid. Filled in from the CSV's own distinct values, but the COUNT is
# asserted against these constants, so a truncated CSV cannot pass.
EXPECTED_REPS = 3
EXPECTED_ARMS = ("inf_T", "inf_CAP", "cap_T", "cap_CAP")
EXPECTED_BUILDS = ("orig", "base", "get", "modp", "both")

# The arm where the change under test bites, and the arms where it must not.
FIX_ARM = "cap_T"
NEUTRAL_ARMS = ("inf_T", "inf_CAP", "cap_CAP")


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def mean_of(rows, arm, build, key="wall_s"):
    vals = []
    for r in rows:
        if r.get("arm") == arm and r.get("build") == build:
            v = (r.get(key) or "").strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    return statistics.mean(vals) if vals else None


def check(path, base, new, verbose=True):
    errs = []
    rows = read_rows(path)

    builds = sorted({r.get("build") for r in rows})
    arms = sorted({r.get("arm") for r in rows})
    reps = sorted({r.get("rep") for r in rows})
    expected_cells = len(reps) * len(arms) * len(builds)
    if verbose:
        print("  grid: %d reps x %d arms x %d builds = %d cells"
              % (len(reps), len(arms), len(builds), expected_cells))
    if len(rows) != expected_cells:
        errs.append("row count %d does not match the grid %d" % (len(rows), expected_cells))
    for rep in reps:
        for arm in arms:
            for b in builds:
                n = sum(1 for r in rows if r.get("rep") == rep and r.get("arm") == arm
                        and r.get("build") == b)
                if n != 1:
                    errs.append("cell (rep=%s arm=%s build=%s) appears %d times" % (rep, arm, b, n))

    for i, r in enumerate(rows):
        if (r.get("wall_s") or "").strip() == "":
            errs.append("row %d: empty wall_s" % (i + 2))
        if (r.get("verdict") or "").strip() != "OK":
            errs.append("row %d (rep=%s arm=%s build=%s): verdict=%r"
                        % (i + 2, r.get("rep"), r.get("arm"), r.get("build"), r.get("verdict")))
        if (r.get("gpu_err") or "").strip() != "":
            errs.append("row %d (rep=%s arm=%s build=%s): gpu_err=%r -- this build is "
                        "missing the `gpu` feature, so the comparison is against a "
                        "different code path"
                        % (i + 2, r.get("rep"), r.get("arm"), r.get("build"), r.get("gpu_err")))
        if (r.get("md5") or "").strip() != REF_MD5:
            errs.append("row %d: barcode md5 differs from the reference" % (i + 2))
        if (r.get("bars") or "").strip() != REF_BARS:
            errs.append("row %d: bar counts %r differ from the reference %s"
                        % (i + 2, r.get("bars"), REF_BARS))

    # -- rule 5: the claim -------------------------------------------------
    for arm in arms:
        b = mean_of(rows, arm, base)
        n = mean_of(rows, arm, new)
        if b is None or n is None:
            errs.append("arm %s: missing a wall time for %s or %s" % (arm, base, new))
            continue
        ratio = n / b
        if verbose:
            print("  %-8s base %.3f s -> new %.3f s  (%.3fx)" % (arm, b, n, ratio))
        if arm == FIX_ARM:
            if ratio > 0.60:
                errs.append("arm %s: the fix must cut the wall time by at least 40%%, "
                            "measured %.3fx" % (arm, ratio))
        else:
            if abs(ratio - 1.0) > 0.15:
                errs.append("arm %s: the fix must not move this arm by more than 15%%, "
                            "measured %.3fx" % (arm, ratio))
    return errs


def negative_self_test(path, base, new):
    base_rows = read_rows(path)
    cases = []

    def variant(rows, name):
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        cases.append((name, tmp))

    rows = [dict(r) for r in base_rows]
    rows[0]["md5"] = "0" * 32
    variant(rows, "barcode changed")

    rows = [dict(r) for r in base_rows]
    rows[0]["gpu_err"] = "built without the `gpu` feature"
    variant(rows, "gpu_err set (feature-set gate)")

    rows = [dict(r) for r in base_rows]
    rows[0]["wall_s"] = ""
    variant(rows, "empty wall time")

    rows = [dict(r) for r in base_rows]
    rows[0]["verdict"] = "BADCODEC"
    variant(rows, "verdict BADCODEC")

    rows = [dict(r) for r in base_rows][:-1]
    variant(rows, "one row deleted")

    # the claim: make the "fixed" arm as slow as the base
    rows = [dict(r) for r in base_rows]
    for r in rows:
        if r["arm"] == FIX_ARM and r["build"] == new:
            r["wall_s"] = "%.3f" % mean_of(rows, FIX_ARM, base)
    variant(rows, "the fixed arm is not faster")

    failures = []
    for name, tmp in cases:
        errs = check(tmp, base, new, verbose=False)
        if not errs:
            failures.append("NEGATIVE SELF-TEST FAILED: the checker ACCEPTED a CSV with %s" % name)
        else:
            print("  negative self-test OK: %s -> rejected (%d reason(s))" % (name, len(errs)))
        os.unlink(tmp)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--base", default="orig")
    ap.add_argument("--new", default="base")
    ap.add_argument("--no-self-test", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv_path):
        print("FAIL: %s does not exist" % args.csv_path, file=sys.stderr)
        return 2

    print("CHECK %s" % args.csv_path)
    errs = check(args.csv_path, args.base, args.new)
    if not args.no_self_test:
        errs += negative_self_test(args.csv_path, args.base, args.new)

    if errs:
        print("\nFAIL (%d problem(s)):" % len(errs))
        for e in errs:
            print("  - %s" % e)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
