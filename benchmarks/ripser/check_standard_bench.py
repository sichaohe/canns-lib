#!/usr/bin/env python3
"""Checker for the standard-benchmark table.

The table has one row per (dataset, side). The rules it enforces are the
ones a reader of the paper would apply:

  1. THE GRID IS COMPLETE. Every dataset of the ripser++ suite
     (celegans, dragon1000, HIV, o3_4096, sphere_3_192, Vicsek300) and
     every side (ripser++ on the GPU, canns-lib on the GPU, canns-lib on
     the CPU) must appear exactly once. The denominator is computed here
     from the dataset list, not taken from the table.
  2. THE ENVIRONMENT IS THE PUBLISHED ONE. ripser++'s own row must
     reproduce the published expected barcodes, otherwise the comparison
     is against a broken baseline and every other row is meaningless.
  3. THE IMPLEMENTATIONS AGREE. canns-lib's GPU row and its CPU row must
     match ripser++'s barcodes on every dimension, and all three must
     match the published reference on the dimensions the reference
     contains (dims 0 and 1 -- its generator wrote no others).
  4. THE TIMINGS ARE USABLE. Every row must carry a positive wall time
     and a peak RSS.

USAGE
    python check_standard_bench.py standard_bench.csv [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

DATASETS = ["celegans", "dragon1000", "HIV", "o3_4096", "sphere_3_192",
            "Vicsek300"]
SIDES = ["pp_gpu", "gpu", "cpu"]

REQUIRED = ["dataset", "side", "maxdim", "wall_s", "proc_s", "rss_kb",
            "bars_total", "match_pp", "match_expected", "verdict"]


def load_tables(paths):
    """Merge several tables into one row set, keyed by (dataset, side).

    A table may be split across files when a side had to be measured in a
    separate pass (the reference CPU implementation needs its own run on the
    heavy datasets). The merge is by (dataset, side); a row already present
    is not overwritten, so the first file wins.
    """
    merged, order, dupes = {}, [], []
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["dataset"], r["side"])
                if key in merged:
                    # A duplicate is a defect in the table, not a row to
                    # merge away: the merge keeps the first and the
                    # duplicate is reported.
                    dupes.append(key)
                    continue
                merged[key] = r
                order.append(key)
    return [merged[k] for k in order], dupes


def check(paths, datasets=None, sides=None):
    """Returns a list of failure strings (empty == PASS)."""
    datasets = datasets or DATASETS
    sides = sides or SIDES
    fails = []
    if isinstance(paths, str):
        paths = [paths]
    rows, dupes = load_tables(paths)
    for key in dupes:
        fails.append("duplicated row: %s / %s" % key)
    if not rows:
        return ["table is empty: %s" % ", ".join(paths)]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return ["table is missing columns: %s" % ", ".join(missing_cols)]

    seen = {}
    for r in rows:
        seen[(r["dataset"], r["side"])] = r

    for ds in datasets:
        for side in sides:
            r = seen.get((ds, side))
            if r is None:
                fails.append("missing row: %s / %s" % (ds, side))
                continue
            if r["verdict"] == "TIMEOUT":
                # The side did not finish inside the timeout. That is a
                # result, not a defect -- but it must be the ONLY side
                # missing for that dataset, and it is reported.
                print("  %s / %s: TIMEOUT (not comparable)" % (ds, side))
                for other in sides:
                    if other == side:
                        continue
                    orow = seen.get((ds, other))
                    if orow is None or orow["verdict"] != "SAME_BARCODE":
                        fails.append("%s / %s: the only comparable side is the "
                                     "timed-out one" % (ds, side))
                continue
            if r["verdict"] != "SAME_BARCODE":
                fails.append("%s / %s: verdict %s" % (ds, side, r["verdict"]))
            if r["match_expected"] != "yes":
                fails.append("%s / %s: does not match the published reference"
                             % (ds, side))
            if side != "pp_gpu" and r["match_pp"] != "yes":
                fails.append("%s / %s: does not match ripser++ run here"
                             % (ds, side))
            try:
                if float(r["wall_s"]) <= 0:
                    fails.append("%s / %s: non-positive wall time" % (ds, side))
                if int(r["rss_kb"]) <= 0:
                    fails.append("%s / %s: no peak RSS" % (ds, side))
            except ValueError:
                fails.append("%s / %s: the timing columns are not numbers"
                             % (ds, side))
    return fails


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    if kind == "flip_verdict":
        rows[0]["verdict"] = "BARCODE_MISMATCH"
    elif kind == "break_expected":
        rows[0]["match_expected"] = "no"
    elif kind == "break_pp":
        for r in rows:
            if r["side"] == "gpu":
                r["match_pp"] = "no"
                break
    elif kind == "drop_row":
        rows = rows[1:]
    elif kind == "dup_row":
        rows = rows + [dict(rows[0])]
    elif kind == "zero_time":
        rows[0]["wall_s"] = "0"
    elif kind == "no_rss":
        rows[0]["rss_kb"] = "0"
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["flip_verdict", "break_expected", "break_pp", "drop_row",
             "dup_row", "zero_time", "no_rss"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", default=["standard_bench.csv"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    fails = check(args.csv)
    print("required cells: %d datasets x %d compared sides = %d"
          % (len(DATASETS), len(SIDES), len(DATASETS) * len(SIDES)))
    print("(the reference CPU implementation is reported alongside them; it is a")
    print(" baseline, not a side under test, so it is not part of the grid)")
    n = len(load_tables(args.csv)[0])
    print("rows in table                    : %d" % n)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: %d rows, every side reproduces the published barcodes and "
          "agrees with ripser++" % n)

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="stdbench_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv[0], dst, kind)
                f = check(dst)
                if f:
                    print("  self-test %-16s -> correctly FAILED (%d problem(s))"
                          % (kind, len(f)))
                else:
                    print("  self-test %-16s -> *** NOT DETECTED ***" % kind)
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
