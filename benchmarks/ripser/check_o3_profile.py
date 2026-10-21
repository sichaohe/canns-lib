#!/usr/bin/env python3
"""Checker for the o3_4096 phase table.

The table holds, per (side, dimension), where the time went on the one
dataset in ripser++'s suite where we lose badly. The claims it has to
support are:

1. **the grid is complete** -- every side has a row for every dimension it
   ran, and the dimensions are the same on both sides;
2. **the filtration is the same** -- the number of columns reduced at each
   dimension agrees between our build and ripser++, because if it does not,
   the two sides are not computing the same thing and no timing comparison
   means anything;
3. **the host was quiet** -- the foreign load on every row is below the
   threshold, or the table is rejected rather than footnoted;
4. **the numbers are internally consistent** -- the per-dimension phases
   add up to something close to the wall time, and a row cannot claim more
   time in its phases than the run took.

It prints its own denominator and carries negative self-tests: a corrupted
copy of the table must make it fail.

USAGE
    python check_o3_profile.py o3_profile.csv [--max-foreign-cpu 0.25]
    python check_o3_profile.py o3_profile.csv --self-test
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

# The sides that have a per-dimension breakdown, and the dimensions each of
# them must have a row for.
SIDES = {
    "ours_gpu": (1, 2, 3),
    "ours_cpu": (1, 2, 3),
    "pp_profiled": (1, 2, 3),
    "pp": (1, 2, 3),
}
# The side whose column counts the other side's must agree with. It is the
# PATCHED copy of ripser++, because the pristine one does not print the
# dimension-1 column count; the pristine one is still measured, as the
# wall-clock reference whose number the paper quotes.
REFERENCE = "pp_profiled"
OURS = "ours_gpu"

REQUIRED = ("side", "dim", "columns", "wall_s", "foreign_cpu", "verdict")


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    by = {}
    for r in rows:
        by.setdefault(r["side"], {})[r["dim"]] = r
    return rows, by


def check(path, max_foreign_cpu=0.25):
    """Returns a list of failure strings (empty == PASS)."""
    fails = []
    rows, by = load(path)
    if not rows:
        return ["the table is empty: %s" % path]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return ["the table is missing columns: %s" % ", ".join(missing_cols)]

    # 1. the grid
    expected = 0
    for side, dims in SIDES.items():
        if side not in by:
            fails.append("no rows for the side %s" % side)
            continue
        for dim in dims:
            expected += 1
            if str(dim) not in by[side]:
                fails.append("missing cell: %s / dim %d" % (side, dim))
        extra = [d for d in by[side] if d.isdigit() and int(d) not in dims]
        for d in extra:
            fails.append("unexpected cell: %s / dim %s" % (side, d))

    # 2. the filtration is the same
    if REFERENCE in by and OURS in by:
        for dim in sorted(set(by[REFERENCE]) & set(by[OURS]), key=lambda d: int(d)):
            a = (by[OURS][dim].get("columns") or "").strip()
            b = (by[REFERENCE][dim].get("columns") or "").strip()
            if not a or not b:
                fails.append("dim %s: the column count is missing on one side "
                             "(ours=%r, reference=%r)" % (dim, a, b))
            elif a != b:
                fails.append("dim %s: the two sides reduced %s and %s columns "
                             "-- they are not computing the same filtration, so "
                             "the timings are not comparable" % (dim, a, b))

    # 3. the host was quiet
    worst_mean, worst_max, worst_row = 0.0, 0.0, None
    for r in rows:
        if (r.get("verdict") or "").strip() != "OK":
            fails.append("%s / dim %s: the run did not succeed (verdict %r)"
                         % (r["side"], r["dim"], r.get("verdict")))
        raw = (r.get("foreign_cpu") or "").strip()
        if not raw:
            fails.append("%s / dim %s: no foreign-load reading, so the table "
                         "cannot show it was measured on a quiet host"
                         % (r["side"], r["dim"]))
            continue
        try:
            value = float(raw)
        except ValueError:
            fails.append("%s / dim %s: the foreign load is not a number (%r)"
                         % (r["side"], r["dim"], raw))
            continue
        if value > worst_mean:
            worst_mean, worst_row = value, "%s / dim %s" % (r["side"], r["dim"])
        try:
            worst_max = max(worst_max, float((r.get("foreign_max") or "0").strip() or 0))
        except ValueError:
            pass
        if max_foreign_cpu > 0 and value > max_foreign_cpu:
            fails.append("%s / dim %s: the host was used by other users at "
                         "%.3f cores, above the %.2f-core threshold"
                         % (r["side"], r["dim"], value, max_foreign_cpu))

    # 4. the phases add up to about the wall time
    for side, dims in by.items():
        wall = 0.0
        for dim in dims:
            try:
                wall = max(wall, float((dims[dim].get("wall_s") or "0").strip() or 0))
            except ValueError:
                pass
        phases = 0.0
        for dim in dims:
            d = dims[dim]
            for col in ("scan_s", "assemble_s", "gpu_scan_s", "reduce_s",
                        "fill_s", "pre_loop_s", "gpu_pre_s", "densify_s"):
                try:
                    phases += float((d.get(col) or "").strip() or 0)
                except ValueError:
                    pass
        if wall > 0 and phases > wall * 1.05:
            fails.append("%s: the per-dimension phases add up to %.3f s but "
                         "the run took %.3f s -- the phases overlap or one of "
                         "them is measured twice" % (side, phases, wall))

    return fails


MUTATIONS = ["drop_row", "change_columns", "busy_host", "no_foreign",
             "fail_verdict", "phases_over_wall"]


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    if kind == "drop_row":
        rows = [r for r in rows if not (r["side"] == "ours_gpu" and r["dim"] == "2")]
    elif kind == "change_columns":
        for r in rows:
            if r["side"] == "ours_gpu" and r["dim"] == "2":
                r["columns"] = str(int(r["columns"]) + 1)
    elif kind == "busy_host":
        for r in rows:
            r["foreign_cpu"] = "1.500"
    elif kind == "no_foreign":
        for r in rows:
            r["foreign_cpu"] = ""
    elif kind == "fail_verdict":
        for r in rows:
            if r["side"] == "pp" and r["dim"] == "1":
                r["verdict"] = "RC=1"
    elif kind == "phases_over_wall":
        for r in rows:
            r["scan_s"] = "999.0"
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="o3_profile.csv")
    ap.add_argument("--max-foreign-cpu", type=float, default=0.25,
                    help="fail if any row was measured while somebody else was "
                         "using more than this many cores; 0 = do not check")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    rows, by = load(args.csv)
    fails = check(args.csv, max_foreign_cpu=args.max_foreign_cpu)
    expected = sum(len(d) for d in SIDES.values())
    print("expected cells : %d sides x their dimensions = %d"
          % (len(SIDES), expected))
    print("rows in table  : %d" % len(rows))
    for side in sorted(by):
        print("  %-10s dims %s" % (side, ",".join(sorted(by[side], key=int))))
    if fails:
        print("\nFAIL (%d problem(s)):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: the grid is complete, the two sides reduced the same number "
          "of columns at every dimension, and every row was measured on a "
          "quiet host")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="o3prof_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, max_foreign_cpu=args.max_foreign_cpu)
                if f:
                    print("  self-test %-18s -> correctly FAILED (%d problem(s))"
                          % (kind, len(f)))
                else:
                    print("  self-test %-18s -> *** NOT DETECTED ***" % kind)
                    bad += 1
            if bad:
                print("\nFAIL: %d mutation(s) slipped through" % bad)
                return 1
            print("\nPASS: all %d negative self-tests detected" % len(MUTATIONS))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
