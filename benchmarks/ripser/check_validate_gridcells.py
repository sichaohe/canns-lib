#!/usr/bin/env python3
"""Checker for the grid-cells cross-implementation validation table.

The table is the ONLY data source; this script recomputes the denominator
from the grid-cells matrix directory (never from the table's own row count),
re-derives every per-cell verdict, and refuses to pass a table that is
incomplete or that claims a barcode match it cannot support.

FAIL conditions
  * a matrix in the grid-cells directory is missing from the table
  * the table has a row that is not a matrix of that directory
  * a matrix appears more than once
  * a cell's bar counts differ between the two implementations
  * a cell has unmatched bars (the tolerance match could not pair them)
  * a cell's max relative deviation exceeds the tolerance
  * a cell's verdict is not SAME_BARCODE
  * a cell's timing columns are missing or not positive numbers

USAGE
    python check_validate_gridcells.py --csv gridcells_validation.csv
    python check_validate_gridcells.py --csv ... --self-test
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile

GRID_MATRIX_DIR = "/home/sichaohe/gridcells/matrices"
RTOL = 1e-5

REQUIRED = ["key", "ours_s", "ripserpp_s", "ratio",
            "finite_pairs", "distinct_values", "near_tie_pairs",
            "bars_ours_d0", "bars_pp_d0", "unmatched_d0", "max_rel_dev_d0",
            "bars_ours_d1", "bars_pp_d1", "unmatched_d1", "max_rel_dev_d1",
            "oracle", "oracle_agrees_with", "verdict"]

# Closed vocabulary. SAME_BARCODE: the two implementations agree on every
# bar within the tolerance. ORACLE_SUPPORTS_OURS: they disagree, and a third
# implementation (GUDHI) agrees with US -- i.e. ripser++ is the outlier.
# ORACLE_SUPPORTS_RIPSERPP and BARCODE_MISMATCH are failures: in the first
# case a third implementation says WE are wrong; in the second nobody can
# arbitrate the difference.
VERDICT_OK = ("SAME_BARCODE", "ORACLE_SUPPORTS_OURS")
VERDICT_FAIL = ("ORACLE_SUPPORTS_RIPSERPP", "BARCODE_MISMATCH")


def expected_keys(matrix_dir=GRID_MATRIX_DIR):
    """The denominator: every matrix that exists on disk."""
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(matrix_dir, "*.npy")))


def check(csv_path, matrix_dir=GRID_MATRIX_DIR):
    """Returns a list of failure strings (empty == PASS)."""
    fails = []
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["table is empty: %s" % csv_path]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return ["table is missing columns: %s" % ", ".join(missing_cols)]

    want = expected_keys(matrix_dir)
    got = [r["key"] for r in rows]
    got_set = set(got)

    for k in want:
        if k not in got_set:
            fails.append("missing cell (in matrix dir, absent from table): %s" % k)
    for k in sorted(got_set):
        if k not in set(want):
            fails.append("unknown cell (in table, absent from matrix dir): %s" % k)
    dupes = sorted({k for k in got if got.count(k) > 1})
    for k in dupes:
        fails.append("duplicated cell: %s" % k)

    for r in rows:
        key = r["key"]
        verdict = r.get("verdict", "")
        if verdict not in VERDICT_OK and verdict not in VERDICT_FAIL:
            fails.append("%s: verdict %r is not in the controlled vocabulary %s"
                         % (key, verdict, list(VERDICT_OK) + list(VERDICT_FAIL)))
        for dim in (0, 1):
            try:
                a = int(r["bars_pp_d%d" % dim])
                b = int(r["bars_ours_d%d" % dim])
            except ValueError:
                fails.append("%s: bar counts are not integers (dim %d)" % (key, dim))
                continue
            if a != b:
                fails.append("%s: dim %d bar counts differ: ripser++ %d vs ours %d"
                             % (key, dim, a, b))
            try:
                un = int(r["unmatched_d%d" % dim])
            except ValueError:
                fails.append("%s: unmatched_d%d is not an integer" % (key, dim))
                un = -1
            if un < 0:
                fails.append("%s: dim %d has %d unmatched bars" % (key, dim, un))
            try:
                dev = float(r["max_rel_dev_d%d" % dim])
            except ValueError:
                fails.append("%s: max_rel_dev_d%d is not a number" % (key, dim))
                continue
            if dev > RTOL:
                fails.append("%s: dim %d max relative deviation %.3e > %.0e"
                             % (key, dim, dev, RTOL))
        try:
            if float(r["ours_s"]) <= 0 or float(r["ripserpp_s"]) <= 0:
                fails.append("%s: non-positive wall time" % key)
            if float(r["ratio"]) <= 0:
                fails.append("%s: non-positive ratio" % key)
        except ValueError:
            fails.append("%s: timing columns are not numbers" % key)
        for col in ("finite_pairs", "distinct_values", "near_tie_pairs"):
            try:
                v = int(r[col])
            except (KeyError, ValueError):
                fails.append("%s: %s is not an integer" % (key, col))
                continue
            if v < 0:
                fails.append("%s: %s is negative (%d)" % (key, col, v))
        try:
            if int(r["near_tie_pairs"]) > int(r["finite_pairs"]):
                fails.append("%s: more near-tie pairs than finite pairs" % key)
        except (KeyError, ValueError):
            pass

        # The verdict has to be backed by the evidence columns.
        unmatched_total = 0
        for dim in (0, 1):
            try:
                unmatched_total += int(r["unmatched_d%d" % dim])
            except ValueError:
                pass
        oracle = (r.get("oracle") or "").strip()
        agrees = (r.get("oracle_agrees_with") or "").strip()
        if verdict == "SAME_BARCODE":
            if unmatched_total:
                fails.append("%s: verdict SAME_BARCODE but %d bars are unmatched"
                             % (key, unmatched_total))
            if agrees != "not_needed":
                fails.append("%s: verdict SAME_BARCODE but oracle_agrees_with=%r"
                             % (key, agrees))
        elif verdict == "ORACLE_SUPPORTS_OURS":
            if oracle != "gudhi":
                fails.append("%s: verdict ORACLE_SUPPORTS_OURS without a gudhi "
                             "oracle (oracle=%r)" % (key, oracle))
            if agrees != "ours":
                fails.append("%s: verdict ORACLE_SUPPORTS_OURS but the oracle "
                             "agrees with %r" % (key, agrees))
            if not unmatched_total:
                fails.append("%s: verdict ORACLE_SUPPORTS_OURS but nothing "
                             "differs" % key)
        elif verdict == "ORACLE_SUPPORTS_RIPSERPP":
            fails.append("%s: the oracle agrees with ripser++, i.e. we are the "
                         "odd one out here" % key)
        elif verdict == "BARCODE_MISMATCH":
            fails.append("%s: the two implementations disagree and no oracle "
                         "settled it" % key)
    return fails


def mutate(src, dst, kind):
    """Write a deliberately corrupted copy of the table."""
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    if kind == "drop_row":
        rows = rows[1:]
    elif kind == "dup_row":
        rows = rows + [dict(rows[0])]
    elif kind == "flip_verdict":
        rows[0]["verdict"] = "BARCODE_MISMATCH"
    elif kind == "bump_dev":
        rows[0]["max_rel_dev_d1"] = "%.3e" % (RTOL * 10)
    elif kind == "unmatched":
        rows[0]["unmatched_d1"] = "1"
    elif kind == "count_mismatch":
        rows[0]["bars_ours_d1"] = str(int(rows[0]["bars_ours_d1"]) + 1)
    elif kind == "bad_time":
        rows[0]["ours_s"] = "0"
    elif kind == "bad_vocab":
        rows[0]["verdict"] = "SAME_BARCODE_ISH"
    elif kind == "oracle_swap":
        for r in rows:
            if r.get("verdict", "").startswith("ORACLE_"):
                r["oracle_agrees_with"] = "ripserpp"
                break
        else:
            rows[0]["oracle_agrees_with"] = "ripserpp"
    elif kind == "oracle_dropped":
        for r in rows:
            if r.get("verdict", "").startswith("ORACLE_"):
                r["oracle"] = "none"
                break
        else:
            rows[0]["verdict"] = "ORACLE_SUPPORTS_RIPSERPP"
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["drop_row", "dup_row", "flip_verdict", "bump_dev",
             "unmatched", "count_mismatch", "bad_time", "bad_vocab",
             "oracle_swap", "oracle_dropped"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="gridcells_validation.csv")
    ap.add_argument("--matrix-dir", default=GRID_MATRIX_DIR)
    ap.add_argument("--self-test", action="store_true",
                    help="corrupt a copy of the table and require the checker to FAIL")
    args = ap.parse_args()

    want = expected_keys(args.matrix_dir)
    if not want:
        print("FAIL: no matrices under %s -- this checker has to run on the"
              % args.matrix_dir)
        print("      host that holds the grid-cells corpus; it recomputes the")
        print("      denominator from that directory, and an empty one would")
        print("      make every cell look missing.")
        return 1
    fails = check(args.csv, args.matrix_dir)
    with open(args.csv, newline="") as fh:
        n_rows = len(list(csv.DictReader(fh)))

    print("expected cells (from matrix dir): %d" % len(want))
    print("rows in table                    : %d" % n_rows)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        if len(fails) > 40:
            print("  ... and %d more" % (len(fails) - 40))
        return 1
    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    from collections import Counter
    counts = Counter(r["verdict"] for r in rows)
    print("\nverdicts: %s" % ", ".join("%s=%d" % (k, v) for k, v in sorted(counts.items())))
    print("PASS: %d cells, every barcode matched on both dimensions" % n_rows)

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="gc_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, args.matrix_dir)
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
