#!/usr/bin/env python3
"""Summarise the load characterisation into the distributions the paper quotes.

The characterisation table has one row per matrix. The paper's first
section needs DISTRIBUTIONS, not one instance: the median and the range
of every property, and the phase decomposition of the wall time (which
is what replaces the "this is not a reduction problem" assertion with a
measurement).

This script reads the table and prints the summary; it does not decide
what the numbers mean.

USAGE
    python summarize_characterization.py characterize_gridcells.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics


def col(rows, key, f=float):
    out = []
    for r in rows:
        v = (r.get(key) or "").strip()
        if not v:
            continue
        try:
            out.append(f(v))
        except ValueError:
            pass
    return out


def stat(name, v, fmt="%.4f"):
    if not v:
        return "%-34s (no data)" % name
    s = sorted(v)
    return "%-34s median %s   min %s   max %s   n=%d" % (
        name, fmt % statistics.median(s), fmt % s[0], fmt % s[-1], len(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="characterize_gridcells.csv")
    args = ap.parse_args()
    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print("matrices: %d" % len(rows))
    print()
    print("-- the data --")
    print(stat("missing-pair fraction", col(rows, "inf_frac")))
    print(stat("finite pairs (median)", col(rows, "finite_pairs"), "%.0f"))
    print()
    print("-- dim 1 --")
    print(stat("columns", col(rows, "dim1_cols"), "%.0f"))
    print(stat("apparent-pair fraction", col(rows, "dim1_apparent_frac"), "%.6f"))
    print(stat("loop seconds", col(rows, "dim1_loop_s"), "%.3f"))
    print()
    print("-- dim 2 --")
    print(stat("columns", col(rows, "dim2_cols"), "%.0f"))
    print(stat("apparent-pair fraction", col(rows, "dim2_apparent_frac"), "%.6f"))
    print(stat("loop seconds", col(rows, "dim2_loop_s"), "%.3f"))
    print(stat("GPU pre-pass seconds", col(rows, "dim2_gpu_pre_s"), "%.3f"))
    print(stat("sharded pivot build seconds", col(rows, "dim2_pivot_build_s"), "%.3f"))
    print()
    print("-- the whole call --")
    print(stat("library wall seconds", col(rows, "ph_wall_s"), "%.3f"))
    print(stat("process wall seconds", col(rows, "wall_s"), "%.3f"))
    print(stat("reduction phase seconds", col(rows, "ph_reduce_s"), "%.3f"))
    print(stat("assembly phase seconds", col(rows, "ph_assemble_s"), "%.3f"))
    print(stat("peak RSS (MB)", [x / 1024.0 for x in col(rows, "rss_kb")], "%.0f"))
    print()
    # the phase shares, per matrix, then their medians
    shares = {"dim1 loop": [], "dim2 loop": [], "GPU pre-pass": [],
              "pivot build": [], "assembly": [], "other": []}
    for r in rows:
        # The LIBRARY wall: the process wall also carries the interpreter
        # start and the .npy load, which are not the implementation's cost.
        wall = (float(r["ph_wall_s"]) if (r.get("ph_wall_s") or "").strip()
                else 0.0)
        if wall <= 0:
            continue
        def g(k):
            try:
                return float(r.get(k) or 0.0)
            except ValueError:
                return 0.0
        d1, d2 = g("dim1_loop_s"), g("dim2_loop_s")
        pre = g("dim2_gpu_pre_s")
        piv = g("dim2_pivot_build_s")
        asm = g("ph_assemble_s")
        red = g("ph_reduce_s")
        other = max(0.0, wall - red - asm)
        shares["dim1 loop"].append(d1 / wall)
        shares["dim2 loop"].append(d2 / wall)
        shares["GPU pre-pass"].append(pre / wall)
        shares["pivot build"].append(piv / wall)
        shares["assembly"].append(asm / wall)
        shares["other"].append(other / wall)
    print("-- the share of the wall time (median over the corpus) --")
    for k, v in shares.items():
        if v:
            print("  %-16s %5.1f%%   (min %4.1f%%  max %4.1f%%)"
                  % (k, 100 * statistics.median(v), 100 * min(v), 100 * max(v)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
