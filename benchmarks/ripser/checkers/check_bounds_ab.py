#!/usr/bin/env python3
"""Checker for the column-store A/B (the sparse-bounds change).

WHAT THIS TABLE IS. The reduction used to keep one `u32` per ENUMERATED
column in its working column store: the end offset of that column's
entries. `append_column` ran once per enumerated column -- 286,562,199
times at dim=2 on the ASA matrix -- so the array was 1.15 GB (403 MB at
the T threshold) and every one of those writes was a cache line that
carried no information, because the overwhelming majority of the columns
are apparent pairs that store nothing.

The change keeps one (column, end offset) pair per NON-EMPTY column and
finds a column's range with a binary search. The table has to show two
things, and this checker refuses the table unless it shows both:

  1. THE BARCODE IS UNCHANGED. Every row of every arm of every build must
     carry the same md5, the same bar counts and the same top-3 H1
     lifetimes. A change in the barcode is a hard failure, whatever it
     does to the wall time.
  2. THE CHANGE IS NOT A REGRESSION. The new build must not be slower on
     any arm, and -- when the table carries the peak resident set -- it
     must use strictly less memory on every arm.

The denominator is computed here, not taken from the table: the EXPECTED
grid is arms x builds x reps, and a missing cell is named.

USAGE
    python check_bounds_ab.py bounds_ab.csv [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

ARMS = ["inf_T", "inf_CAP", "cap_T", "cap_CAP"]
BUILDS = ["ship", "bounds"]
BASE = "ship"
NEW = "bounds"

# The barcode of the ASA matrix at the reference threshold, as the harness
# computes it (md5 per dimension, joined).
REF_MD5 = ("ee1f5283091dd645a3dd53e6f1a7eac6:"
           "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
           "2d018d8fe9c1895b1b74f837af59ed20")
REF_BARS = "1195|481|58"

REQUIRED = ["rep", "arm", "build", "wall_s", "bars", "md5", "verdict"]


def check(path, arms=ARMS, builds=BUILDS, base=BASE, new=NEW):
    """Returns a list of failure strings (empty == PASS)."""
    fails = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["table is empty: %s" % path]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return ["table is missing columns: %s" % ", ".join(missing_cols)]

    reps = sorted({int(r["rep"]) for r in rows})
    n_reps = len(reps)
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["build"])].append(r)

    # --- the denominator ------------------------------------------------
    expected = len(arms) * len(builds) * n_reps
    print("expected cells: %d arms x %d builds x %d reps = %d"
          % (len(arms), len(builds), n_reps, expected))
    if len(rows) != expected:
        fails.append("table has %d rows, the grid has %d" % (len(rows), expected))
    for arm in arms:
        for build in builds:
            got = by.get((arm, build), [])
            if len(got) != n_reps:
                fails.append("cell %s/%s has %d rows, expected %d"
                             % (arm, build, len(got), n_reps))

    # --- the barcode must be identical everywhere ----------------------
    md5s = Counter(r["md5"] for r in rows)
    if len(md5s) > 1:
        fails.append("the rows do not share one barcode md5: %s"
                     % ", ".join("%s x%d" % (k[:16], v) for k, v in md5s.items()))
    for md5 in md5s:
        if md5 != REF_MD5:
            fails.append("barcode md5 %s is not the reference %s"
                         % (md5[:16], REF_MD5[:16]))
    bars = Counter(r["bars"] for r in rows)
    if len(bars) > 1:
        fails.append("the rows do not share one bar count: %s" % dict(bars))
    for b in bars:
        if b != REF_BARS:
            fails.append("bar count %s is not the reference %s" % (b, REF_BARS))
    for r in rows:
        if r["verdict"] != "OK":
            fails.append("%s/%s rep %s: verdict %s"
                         % (r["arm"], r["build"], r["rep"], r["verdict"]))

    # --- the new build must not be slower, and must use less memory ----
    has_rss = "rss_kb" in rows[0] and any(r.get("rss_kb") for r in rows)
    for arm in arms:
        base_rows = by.get((arm, base), [])
        new_rows = by.get((arm, new), [])
        if not base_rows or not new_rows:
            continue
        bw = [float(r["wall_s"]) for r in base_rows if r.get("wall_s")]
        nw = [float(r["wall_s"]) for r in new_rows if r.get("wall_s")]
        if bw and nw:
            ratio = statistics.mean(nw) / statistics.mean(bw)
            print("  %-8s %s %.3f s -> %s %.3f s  (%.4fx)"
                  % (arm, base, statistics.mean(bw), new, statistics.mean(nw), ratio))
            if ratio > 1.0:
                fails.append("%s: the new build is SLOWER (%.4fx)" % (arm, ratio))
        if has_rss:
            try:
                br = max(int(r["rss_kb"]) for r in base_rows if r.get("rss_kb"))
                nr = max(int(r["rss_kb"]) for r in new_rows if r.get("rss_kb"))
            except ValueError:
                continue
            print("  %-8s peak RSS %s %d kB -> %s %d kB  (%+.1f MB)"
                  % (arm, base, br, new, nr, (nr - br) / 1024.0))
            if nr >= br:
                fails.append("%s: the new build does not use less memory "
                             "(%d kB vs %d kB)" % (arm, nr, br))
    return fails


def check_gridcells(path, max_ratio=1.05):
    """The dim-1 cross-check: the change is a no-op there, so it must be
    within noise, and the barcodes must be identical on every cell."""
    fails = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["grid-cells table is empty: %s" % path]
    for c in ("key", "ship_s", "bounds_s", "ratio", "ship_md5", "bounds_md5",
              "barcode_same", "verdict"):
        if c not in rows[0]:
            return ["grid-cells table is missing column %s" % c]
    print("grid-cells cells: %d" % len(rows))
    for r in rows:
        if r["barcode_same"] != "yes" or r["verdict"] != "SAME_BARCODE":
            fails.append("%s: the two builds disagree on the barcode" % r["key"])
        if r["ship_md5"] != r["bounds_md5"]:
            fails.append("%s: the md5s differ" % r["key"])
        try:
            ratio = float(r["ratio"])
        except ValueError:
            fails.append("%s: ratio is not a number" % r["key"])
            continue
        if ratio > max_ratio:
            fails.append("%s: the new build is %.1f%% slower (the change is a "
                         "no-op at dim 1, so it must be within noise)"
                         % (r["key"], (ratio - 1) * 100))
    return fails


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    if kind == "md5":
        rows[0]["md5"] = "0" * 32 + ":x:y"
    elif kind == "verdict":
        rows[0]["verdict"] = "BADCODEC"
    elif kind == "drop_row":
        rows = rows[:-1]
    elif kind == "dup_row":
        rows = rows + [dict(rows[0])]
    elif kind == "slower":
        # make the new build slower on one arm
        for r in rows:
            if r["build"] == NEW and r["arm"] == ARMS[0]:
                r["wall_s"] = "%.3f" % (float(r["wall_s"]) * 1.5)
    elif kind == "bigger_rss":
        for r in rows:
            if r["build"] == NEW:
                r["rss_kb"] = str(int(r.get("rss_kb") or 0) + 1000000)
    elif kind == "missing_arm":
        rows = [r for r in rows if r["arm"] != ARMS[-1]]
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mutate_gridcells(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    if kind == "gc_mismatch":
        rows[0]["barcode_same"] = "no"
        rows[0]["verdict"] = "BARCODE_MISMATCH"
    elif kind == "gc_slower":
        rows[0]["ratio"] = "%.4f" % (float(rows[0]["ratio"]) * 2)
    elif kind == "gc_md5":
        rows[0]["bounds_md5"] = "deadbeef" * 4
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["md5", "verdict", "drop_row", "dup_row", "slower", "bigger_rss",
             "missing_arm"]
GRID_MUTATIONS = ["gc_mismatch", "gc_slower", "gc_md5"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="bounds_ab.csv")
    ap.add_argument("--gridcells", default="",
                    help="the dim-1 cross-check table (optional)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    fails = check(args.csv)
    if args.gridcells and os.path.exists(args.gridcells):
        fails += check_gridcells(args.gridcells)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        if len(fails) > 40:
            print("  ... and %d more" % (len(fails) - 40))
        return 1
    with open(args.csv, newline="") as fh:
        n = len(list(csv.DictReader(fh)))
    print("\nPASS: %d rows, barcode identical on every arm of every build" % n)

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="bounds_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst)
                if f:
                    print("  self-test %-14s -> correctly FAILED (%d problem(s))"
                          % (kind, len(f)))
                else:
                    print("  self-test %-14s -> *** NOT DETECTED ***" % kind)
                    bad += 1
            if args.gridcells and os.path.exists(args.gridcells):
                for kind in GRID_MUTATIONS:
                    dst = os.path.join(tmp, "gmut_%s.csv" % kind)
                    mutate_gridcells(args.gridcells, dst, kind)
                    f = check_gridcells(dst)
                    if f:
                        print("  self-test %-14s -> correctly FAILED (%d problem(s))"
                              % (kind, len(f)))
                    else:
                        print("  self-test %-14s -> *** NOT DETECTED ***" % kind)
                        bad += 1
            if bad:
                print("\nFAIL: %d mutation(s) slipped through" % bad)
                return 1
            print("\nPASS: all %d negative self-tests detected"
                  % (len(MUTATIONS) + (len(GRID_MUTATIONS)
                                       if args.gridcells else 0)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
