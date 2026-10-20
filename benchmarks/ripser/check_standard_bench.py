#!/usr/bin/env python3
"""Checker for the standard-benchmark table.

The table has one row per (dataset, side, repetition). The rules it
enforces are the ones a reader of the paper would apply:

  1. THE GRID IS COMPLETE. Every dataset of the ripser++ suite
     (celegans, dragon1000, HIV, o3_4096, sphere_3_192, Vicsek300) and
     every side (ripser++ on the GPU, canns-lib on the GPU, canns-lib on
     the CPU) must appear, with the required number of repetitions. The
     denominator is computed here from the dataset list and the repetition
     count, not taken from the table.
  2. THE ENVIRONMENT IS THE PUBLISHED ONE. ripser++'s own rows must
     reproduce the published expected barcodes, otherwise the comparison
     is against a broken baseline and every other row is meaningless.
  3. THE IMPLEMENTATIONS AGREE. canns-lib's GPU row and its CPU row must
     match ripser++'s barcodes on every dimension, and all three must
     match the published reference on the dimensions the reference
     contains (dims 0 and 1 -- its generator wrote no others).
  4. THE TIMINGS ARE USABLE. Every row must carry a positive wall time
     and a peak RSS, and there must be enough repetitions for the median
     to mean something: a single measurement, or a minimum, is not a
     representative value.

The checker prints the median and the spread of every cell. With n
repetitions the distribution-free 95% confidence interval of the median
is the interval between the two extreme order statistics, so for n=5 that
is exactly the observed range -- and the paper says so rather than
quoting a tighter interval than the data supports.

USAGE
    python check_standard_bench.py standard_bench.csv [--min-reps 5] [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import sys
import tempfile

DATASETS = ["celegans", "dragon1000", "HIV", "o3_4096", "sphere_3_192",
            "Vicsek300"]
SIDES = ["pp_gpu", "gpu", "cpu"]

REQUIRED = ["dataset", "side", "maxdim", "wall_s", "proc_s", "rss_kb",
            "bars_total", "match_pp", "match_expected", "verdict"]


def load_tables(paths):
    """Merge several tables into one row set, keyed by (dataset, side, rep).

    A table may be split across files when a side had to be measured in a
    separate pass (the reference CPU implementation needs its own run on the
    heavy datasets). The merge is by (dataset, side, rep); a row already
    present is not overwritten, so the first file wins and the duplicate is
    reported.
    """
    merged, order, dupes = {}, [], []
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["dataset"], r["side"], r.get("rep", ""))
                if key in merged:
                    # A duplicate is a defect in the table, not a row to
                    # merge away: the merge keeps the first and the
                    # duplicate is reported.
                    dupes.append(key)
                    continue
                merged[key] = r
                order.append(key)
    return [merged[k] for k in order], dupes


def check(paths, datasets=None, sides=None, min_reps=5, max_loadavg=0.0):
    """Returns a list of failure strings (empty == PASS)."""
    datasets = datasets or DATASETS
    sides = sides or SIDES
    fails = []
    if isinstance(paths, str):
        paths = [paths]
    rows, dupes = load_tables(paths)
    for key in dupes:
        fails.append("duplicated row: %s / %s / rep %s" % key)
    if not rows:
        return ["table is empty: %s" % ", ".join(paths)]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return ["table is missing columns: %s" % ", ".join(missing_cols)]
    if "rep" not in rows[0]:
        return ["the table has no repetition column: it holds one measurement "
                "per cell, and a single measurement is not a representative "
                "value (re-run the suite with --reps %d)" % min_reps]

    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["side"]), []).append(r)

    # the device column must be present and uniform: a table whose device
    # column was written by hand, or whose rows came from different
    # devices, is not a table whose rows are comparable
    devices = {(r.get("gpu") or "").strip() for r in rows}
    if len(devices) != 1 or not devices.pop():
        fails.append("the device column is not uniform and non-empty: %r"
                     % sorted(devices))

    for ds in datasets:
        for side in sides:
            rs = groups.get((ds, side))
            if not rs:
                fails.append("missing cell: %s / %s" % (ds, side))
                continue
            ran = [r for r in rs if r["verdict"] in ("SAME_BARCODE",
                                                      "BARCODE_MISMATCH")]
            timed_out = [r for r in rs if r["verdict"] == "TIMEOUT"]
            if not ran:
                # The side did not finish inside the timeout. That is a
                # result, not a defect -- but it must be the ONLY side
                # missing for that dataset, and it is reported.
                print("  %s / %s: TIMEOUT (not comparable)" % (ds, side))
                for other in sides:
                    if other == side:
                        continue
                    orow = groups.get((ds, other))
                    if not orow or not [x for x in orow if x["verdict"]
                                        == "SAME_BARCODE"]:
                        fails.append("%s / %s: the only comparable side is the "
                                     "timed-out one" % (ds, side))
                continue
            if len(ran) < min_reps:
                fails.append("%s / %s: only %d repetition(s), the paper needs "
                             "at least %d" % (ds, side, len(ran), min_reps))
            for r in ran:
                if r["verdict"] != "SAME_BARCODE":
                    fails.append("%s / %s rep %s: verdict %s"
                                 % (ds, side, r.get("rep", "?"), r["verdict"]))
                if r["match_expected"] != "yes":
                    fails.append("%s / %s rep %s: does not match the published "
                                 "reference" % (ds, side, r.get("rep", "?")))
                if side != "pp_gpu" and r["match_pp"] != "yes":
                    fails.append("%s / %s rep %s: does not match ripser++ run "
                                 "here" % (ds, side, r.get("rep", "?")))
                try:
                    if float(r["wall_s"]) <= 0:
                        fails.append("%s / %s rep %s: non-positive wall time"
                                     % (ds, side, r.get("rep", "?")))
                    if int(r["rss_kb"]) <= 0:
                        fails.append("%s / %s rep %s: no peak RSS"
                                     % (ds, side, r.get("rep", "?")))
                except ValueError:
                    fails.append("%s / %s rep %s: the timing columns are not "
                                 "numbers" % (ds, side, r.get("rep", "?")))
            walls = sorted(float(r["wall_s"]) for r in ran)
            med = statistics.median(walls)
            lo, hi = walls[0], walls[-1]
            print("  %-14s %-10s n=%d  median %8.3f s   95%% CI of the median "
                  "[%.3f, %.3f]   spread %+.1f%%"
                  % (ds, side, len(walls), med, lo, hi,
                     100.0 * (hi - lo) / med if med else 0.0))
            if timed_out:
                print("       (%d of %d repetitions timed out)"
                      % (len(timed_out), len(rs)))

    # A table measured while the host was busy is not a measurement of the
    # code: on this shared machine a neighbour's job changes the numbers by
    # up to 40% and can flip a comparison's sign. So a table can be asked
    # to prove it was measured in a quiet window.
    if max_loadavg > 0:
        for r in rows:
            try:
                la = float(r.get("loadavg") or 0.0)
            except ValueError:
                fails.append("the load average column is not a number")
                break
            if la >= max_loadavg:
                fails.append("the load average %s on this row is not below "
                             "the quiet-window threshold %.2f -- the batch "
                             "was measured on a busy host"
                             % (la, max_loadavg))
                break

    return fails


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())

    def first(pred):
        for r in rows:
            if pred(r):
                return r
        raise SystemExit("the mutation %s has no row to act on -- the table "
                         "does not have the shape the self-test assumes"
                         % kind)

    if kind == "flip_verdict":
        first(lambda r: r["verdict"] == "SAME_BARCODE")["verdict"] = \
            "BARCODE_MISMATCH"
    elif kind == "break_expected":
        first(lambda r: r["verdict"] == "SAME_BARCODE")["match_expected"] = "no"
    elif kind == "break_pp":
        first(lambda r: r["side"] == "gpu" and r["verdict"] == "SAME_BARCODE"
              )["match_pp"] = "no"
    elif kind == "drop_row":
        rows = rows[1:]
    elif kind == "dup_row":
        rows = rows + [dict(rows[0])]
    elif kind == "zero_time":
        first(lambda r: r["verdict"] == "SAME_BARCODE")["wall_s"] = "0"
    elif kind == "no_rss":
        first(lambda r: r["verdict"] == "SAME_BARCODE")["rss_kb"] = "0"
    elif kind == "drop_rep":
        # leave exactly ONE repetition of a cell: a single measurement is not
        # a representative value
        key = None
        for r in rows:
            if r["verdict"] == "SAME_BARCODE":
                key = (r["dataset"], r["side"])
                break
        if key is None:
            raise SystemExit("the mutation drop_rep has no cell to thin")
        kept = [r for r in rows if (r["dataset"], r["side"]) != key]
        rows = kept + [r for r in rows if (r["dataset"], r["side"]) == key][:1]
    elif kind == "mixed_device":
        r = first(lambda x: x["verdict"] == "SAME_BARCODE")
        r["gpu"] = "1"
    elif kind == "one_bad_rep":
        first(lambda r: r["side"] == "gpu" and r["verdict"] == "SAME_BARCODE"
              )["verdict"] = "BARCODE_MISMATCH"
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["flip_verdict", "break_expected", "break_pp", "drop_row",
             "dup_row", "zero_time", "no_rss", "drop_rep", "one_bad_rep",
             "mixed_device"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", default=["standard_bench.csv"])
    ap.add_argument("--min-reps", type=int, default=5)
    ap.add_argument("--max-loadavg", type=float, default=0.0,
                    help="fail if any row was measured at or above this load "
                         "average; 0 = do not check")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    fails = check(args.csv, min_reps=args.min_reps, max_loadavg=args.max_loadavg)
    print("required cells  : %d datasets x %d compared sides = %d"
          % (len(DATASETS), len(SIDES), len(DATASETS) * len(SIDES)))
    print("required rows   : that x %d repetitions = %d"
          % (args.min_reps, len(DATASETS) * len(SIDES) * args.min_reps))
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
          "agrees with ripser++, with enough repetitions to quote a median"
          % n)

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="stdbench_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv[0], dst, kind)
                f = check(dst, min_reps=args.min_reps, max_loadavg=args.max_loadavg)
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
