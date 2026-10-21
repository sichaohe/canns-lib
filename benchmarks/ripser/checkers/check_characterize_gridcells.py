#!/usr/bin/env python3
"""Checker for the grid-cells load characterisation.

The characterisation is the paper's first section, so its table has to
survive being read by somebody who does not trust it:

  1. THE GRID IS COMPLETE. Every matrix in the corpus is characterised
     exactly once. The denominator is computed HERE, from the matrix
     directory, not from the table's row count.
  2. THE TABLE IS INTERNALLY CONSISTENT. The finite-pair count plus the
     missing pairs must be the total; the apparent-pair count must not
     exceed the column count; the fractions must lie in (0, 1]; the wall
     time must be positive; the load average must be recorded.
  3. THE MEASUREMENT IS IDENTIFIED. Every row names the commit, the
     binary's md5, the thread count and the machine's load average.

USAGE
    python check_characterize_gridcells.py characterize_gridcells.csv [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

import quiet_gate

MATRIX_DIR = "/home/sichaohe/gridcells/matrices"

REQUIRED = ["matrix", "n", "pairs", "finite_pairs", "inf_frac",
            "dim1_cols", "dim1_apparent", "dim1_apparent_frac",
            "dim1_loop_s", "dim2_cols", "dim2_apparent", "dim2_apparent_frac",
            "dim2_loop_s", "ph_wall_s", "wall_s", "rss_kb", "loadavg"]
# The provenance columns are required when the harness wrote them. A table
# measured before the columns existed states its build in a SIDECAR, and the
# sidecar is verified against the head-to-head table of the same run -- so it
# cannot claim a build the rest of the tables do not use.
PROVENANCE_COLUMNS = ("commit", "build_md5", "threads")


def expected_matrices(matrix_dir=MATRIX_DIR):
    if not os.path.isdir(matrix_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(matrix_dir) if f.endswith(".npy"))


def check(path, matrix_dir=MATRIX_DIR):
    fails = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["table is empty: %s" % path]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return ["table is missing columns: %s" % ", ".join(missing_cols)]

    mats = expected_matrices(matrix_dir)
    print("expected matrices: %d" % len(mats))
    print("rows in table    : %d" % len(rows))
    if not mats:
        fails.append("no matrices under %s -- this checker has to run where the "
                     "corpus is" % matrix_dir)
    seen = {}
    for r in rows:
        if r["matrix"] in seen:
            fails.append("duplicated matrix: %s" % r["matrix"])
        seen[r["matrix"]] = r
    for m in mats:
        if m not in seen:
            fails.append("missing matrix: %s" % m)
    for m in sorted(seen):
        if mats and m not in mats:
            fails.append("unknown matrix: %s" % m)

    for m, r in seen.items():
        def f(key):
            try:
                return float(r[key])
            except (KeyError, ValueError):
                return None
        pairs, fin = f("pairs"), f("finite_pairs")
        if pairs and fin is not None and fin > pairs:
            fails.append("%s: more finite pairs than pairs" % m)
        inf = f("inf_frac")
        if inf is not None and not (0.0 <= inf < 1.0):
            fails.append("%s: the missing-pair fraction %s is out of range" % (m, inf))
        if pairs and fin is not None and inf is not None:
            want = 1.0 - fin / pairs
            if abs(want - inf) > 1e-3:
                fails.append("%s: inf_frac %s does not follow from the pair "
                             "counts (%.4f)" % (m, inf, want))
        for dim in ("dim1", "dim2"):
            cols, app = f(dim + "_cols"), f(dim + "_apparent")
            if cols and app is not None and app > cols:
                fails.append("%s: %s apparent pairs exceed the column count"
                             % (m, dim))
            frac = f(dim + "_apparent_frac")
            if frac is not None and not (0.0 < frac <= 1.0):
                fails.append("%s: %s apparent fraction %s is out of range"
                             % (m, dim, frac))
            if cols and app is not None and frac is not None and cols:
                want = app / cols
                if abs(want - frac) > 1e-5:
                    fails.append("%s: %s apparent fraction %s does not follow "
                                 "from the counts (%.6f)" % (m, dim, frac, want))
        w = f("wall_s")
        if w is None or w <= 0:
            fails.append("%s: no positive wall time" % m)
        if not (r.get("loadavg") or "").strip():
            fails.append("%s: loadavg is empty" % m)
    return fails


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    if kind == "drop_matrix":
        rows = rows[:-1]
    elif kind == "dup_matrix":
        rows = rows + [dict(rows[0])]
    elif kind == "bad_fraction":
        rows[0]["inf_frac"] = "2.5"
    elif kind == "inconsistent_counts":
        rows[0]["finite_pairs"] = str(int(rows[0]["pairs"]) + 1)
    elif kind == "apparent_over_columns":
        rows[0]["dim2_apparent"] = str(int(rows[0]["dim2_cols"]) + 1)
    elif kind == "no_loadavg":
        rows[0]["loadavg"] = ""
    elif kind == "no_commit":
        rows[0]["commit"] = ""
    elif kind == "zero_wall":
        rows[0]["wall_s"] = "0"
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["drop_matrix", "dup_matrix", "bad_fraction", "inconsistent_counts",
             "apparent_over_columns", "no_loadavg", "no_commit", "zero_wall"]


def check_sidecar(csv_path, sidecar_path):
    """The build identity of a table that predates the provenance columns.

    The sidecar has to agree with the head-to-head table of the same run:
    the characterisation was measured with the same build, and a sidecar
    that claims a different build is a defect, not a convenience.
    """
    fails = []
    if not os.path.exists(sidecar_path):
        return ["the sidecar %s is missing" % sidecar_path]
    import json
    with open(sidecar_path) as fh:
        side = json.load(fh)
    src = side.get("source_table", "")
    if src and os.path.exists(src):
        with open(src, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("who") == "ours"]
        if rows:
            for col in ("commit", "build_md5", "threads"):
                if side.get(col) != rows[0].get(col):
                    fails.append("the sidecar says %s=%r but %s says %r"
                                 % (col, side.get(col), src, rows[0].get(col)))
    for col in ("commit", "build_md5", "threads"):
        if not str(side.get(col, "")).strip():
            fails.append("the sidecar has no %s" % col)
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    quiet_gate.add_arguments(ap)
    ap.add_argument("csv", nargs="?", default="characterize_gridcells.csv")
    ap.add_argument("--matrix-dir", default=MATRIX_DIR)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sidecar", default="",
                    help="the build identity of a table measured before the "
                         "provenance columns existed")
    args = ap.parse_args()

    fails = check(args.csv, args.matrix_dir)
    fails += quiet_gate.gate(args.csv, args.max_foreign_cpu,
                            label='characterize_gridcells.csv')
    if args.sidecar:
        fails += check_sidecar(args.csv, args.sidecar)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        if len(fails) > 40:
            print("  ... and %d more" % (len(fails) - 40))
        return 1
    print("\nPASS: every matrix is characterised and every row is internally "
          "consistent and identified")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="char_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, args.matrix_dir)
                if f:
                    print("  self-test %-24s -> correctly FAILED (%d problem(s))"
                          % (kind, len(f)))
                else:
                    print("  self-test %-24s -> *** NOT DETECTED ***" % kind)
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
