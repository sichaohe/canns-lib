#!/usr/bin/env python3
"""Checker for the reduction-split table.

The table holds, per (side, dimension), the two implementations' reduction
on the same filtration: the columns they reduced, the reduction rounds they
took, the peak coboundary heap, and the most rounds any single column took.

**The four count columns must be IDENTICAL between the two sides.** That is
the whole point of the table: if the counts agree, the two implementations
are doing the same work, and the difference between their wall clocks is a
difference in constants rather than a difference in algorithm. If any count
differs, the table is not evidence of anything, and this checker rejects it.

The time columns are NOT required to agree -- they are the thing being
compared, and they are expected to differ.

It derives the grid from the table (which sides and which dimensions are
present), it requires every cell to be filled, it requires the counts to be
positive integers, and it carries negative self-tests: a copy of the table
in which one side's count has been changed must be rejected.

USAGE
    python check_reduce_split.py reduce_split.csv
    python check_reduce_split.py reduce_split.csv --self-test
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

import quiet_gate

# The sides that must be present.
SIDES = ("ours", "ripserpp")

# The columns that have to match, and the columns that are allowed to
# differ.
MUST_MATCH = ("cols", "rounds", "heap_max", "col_rounds_max")
TIME_COLUMNS = ("scan_s", "fill_s", "reduce_s", "total_s")
REQUIRED = ("side", "dim") + MUST_MATCH + TIME_COLUMNS


def check(path, verbose=True):
    """Returns a list of failure strings (empty == PASS)."""
    errs = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["the table is empty: %s" % path]
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        return ["the table is missing columns: %s" % ", ".join(missing)]

    by = {}
    for r in rows:
        key = (r["side"], r["dim"])
        if key in by:
            errs.append("duplicate row: side %s dim %s" % key)
        by[key] = r

    sides = sorted({r["side"] for r in rows})
    for side in SIDES:
        if side not in sides:
            errs.append("the side %s is missing (the table has %s)"
                        % (side, ", ".join(sides)))
    if errs:
        return errs

    dims = sorted({r["dim"] for r in rows if r["side"] == SIDES[0]},
                  key=lambda d: int(d) if d.isdigit() else 0)
    if not dims:
        return ["the table has no rows for the side %s" % SIDES[0]]
    for dim in dims:
        for side in SIDES:
            if (side, dim) not in by:
                errs.append("missing cell: %s / dim %s" % (side, dim))
    if errs:
        return errs

    for dim in dims:
        o, p = by[(SIDES[0], dim)], by[(SIDES[1], dim)]
        for col in MUST_MATCH:
            if o[col] != p[col]:
                errs.append(
                    "dim %s: %s differs -- ours %s, ripser++ %s. The two "
                    "implementations are NOT doing the same work, so the "
                    "wall-clock difference between them cannot be called a "
                    "constant factor." % (dim, col, o[col], p[col]))
            try:
                if int(o[col]) <= 0:
                    errs.append("dim %s: %s is %r, which is not a positive "
                                "count" % (dim, col, o[col]))
            except ValueError:
                errs.append("dim %s: %s is %r, which is not a number"
                            % (dim, col, o[col]))
        if verbose:
            try:
                ot, pt = float(o["total_s"]), float(p["total_s"])
                if ot > 0 and pt > 0:
                    print("  dim %s: ours %.3f s vs ripser++ %.3f s (%.3fx) -- "
                          "same work, different constants"
                          % (dim, ot, pt, ot / pt))
            except (ValueError, KeyError):
                pass
    return errs


def mutate(src, dst, kind):
    """A corrupted copy of the table, for the negative self-test."""
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    if kind == "one_more_round":
        for r in rows:
            if r["side"] == "ours":
                r["rounds"] = str(int(r["rounds"]) + 1)
                break
    elif kind == "different_columns":
        for r in rows:
            if r["side"] == "ours":
                r["cols"] = str(int(r["cols"]) + 1)
                break
    elif kind == "zero_heap":
        for r in rows:
            if r["side"] == "ripserpp":
                r["heap_max"] = "0"
                break
    elif kind == "drop_a_row":
        rows = [r for r in rows if not (r["side"] == "ripserpp" and r["dim"] == "2")]
    elif kind == "drop_a_side":
        rows = [r for r in rows if r["side"] != "ripserpp"]
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ("one_more_round", "different_columns", "zero_heap", "drop_a_row",
             "drop_a_side")


def main() -> int:
    ap = argparse.ArgumentParser()
    quiet_gate.add_arguments(ap)
    ap.add_argument("csv", nargs="?", default="reduce_split.csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    errs = check(args.csv)
    errs += quiet_gate.gate(args.csv, args.max_foreign_cpu,
                            label='reduce_split.csv')
    rows = list(csv.DictReader(open(args.csv, newline="")))
    dims = sorted({r["dim"] for r in rows})
    print("expected cells : %d sides x %d dimensions = %d"
          % (len(SIDES), len(dims), len(SIDES) * len(dims)))
    print("rows in table  : %d" % len(rows))
    if errs:
        print("\nFAIL (%d problem(s)):" % len(errs))
        for e in errs[:40]:
            print("  - %s" % e)
        return 1
    print("\nPASS: both sides reduced the same number of columns, took the "
          "same number of rounds, and reached the same peak coboundary heap "
          "at every dimension -- the same work, different constants")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="reduce_split_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, verbose=False)
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
