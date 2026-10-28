#!/usr/bin/env python3
"""Merge two grid-cells validation tables into a build-vs-build comparison.

WHY. The column-store change (one bound per NON-EMPTY column instead of one
per enumerated column) only has anything to save at dim >= 2: the grid-cells
validation runs at maxdim=1, where the reduction walks the EDGE columns
(~500k of them), so its bounds array was 2 MB and the change is a no-op
there. This table is the evidence for that claim -- the same 12 cells, the
same data, the two builds -- so "the change does not regress another
dataset" is a measured statement rather than an assumption.

The two input tables come from `validate_gridcells.py`, which compares
each build against ripser++ cell by cell. This script pairs them up and
keeps both sides' barcodes, so the comparison can also be checked for
correctness, not just for speed.

USAGE
    python bounds_gridcells_ab.py --a gc_ship.csv --b gc_bounds.csv \
        --out bounds_gridcells_ab.csv
"""
from __future__ import annotations

import argparse
import csv


def load(path):
    with open(path, newline="") as fh:
        return {r["key"]: r for r in csv.DictReader(fh)}


def worst_foreign(rows):
    """The largest foreign-load reading among the given rows (0.0 if none)."""
    out = 0.0
    for r in rows:
        try:
            out = max(out, float(r.get("foreign_cpu") or 0.0))
        except (TypeError, ValueError):
            pass
    return out


def worst_foreign_max(rows):
    """The largest foreign-load peak among the given rows (0.0 if none)."""
    out = 0.0
    for r in rows:
        try:
            out = max(out, float(r.get("foreign_max") or 0.0))
        except (TypeError, ValueError):
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the baseline build's table")
    ap.add_argument("--b", required=True, help="the new build's table")
    ap.add_argument("--out", default="bounds_gridcells_ab.csv")
    ap.add_argument("--max-ratio", type=float, default=1.05,
                    help="the new build may not be more than this much slower "
                         "on any cell (it is a no-op at dim 1, so it must be "
                         "within noise)")
    args = ap.parse_args()

    a = load(args.a)
    b = load(args.b)
    keys = [k for k in a if k in b]
    if not keys:
        print("FAIL: no cell in common", file=__import__("sys").stderr)
        return 2

    rows = []
    for k in keys:
        ra, rb = a[k], b[k]
        sa, sb = float(ra["ours_s"]), float(rb["ours_s"])
        same = (ra["h1_md5_ours"] == rb["h1_md5_ours"]
                and ra["bars_ours_d0"] == rb["bars_ours_d0"]
                and ra["bars_ours_d1"] == rb["bars_ours_d1"]
                and ra["unmatched_d0"] == rb["unmatched_d0"]
                and ra["unmatched_d1"] == rb["unmatched_d1"])
        rows.append({
            "key": k,
            "ship_s": "%.3f" % sa,
            "bounds_s": "%.3f" % sb,
            "ratio": "%.4f" % (sb / sa),
            "ship_md5": ra["h1_md5_ours"],
            "bounds_md5": rb["h1_md5_ours"],
            "barcode_same": "yes" if same else "no",
            "verdict": "SAME_BARCODE" if same else "BARCODE_MISMATCH",
            # The quiet-window reading is carried over from the two tables
            # this summary compares, so the summary can be gated too: the
            # worst reading of either side is the worst reading of the
            # comparison.
            "foreign_cpu": "%.4f" % worst_foreign([ra, rb]),
            "foreign_max": "%.3f" % worst_foreign_max([ra, rb]),
        })
        print("%-46s %8.3f -> %8.3f  %.4f  %s"
              % (k, sa, sb, sb / sa, rows[-1]["verdict"]))

    fields = ["key", "ship_s", "bounds_s", "ratio", "ship_md5", "bounds_md5",
              "barcode_same", "verdict", "foreign_cpu", "foreign_max"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s (%d cells)" % (args.out, len(rows)))
    bad = [r for r in rows if r["verdict"] != "SAME_BARCODE"]
    slow = [r for r in rows if float(r["ratio"]) > args.max_ratio]
    print("barcode mismatches: %d" % len(bad))
    print("cells more than %.0f%% slower: %d" % ((args.max_ratio - 1) * 100, len(slow)))
    return 1 if (bad or slow) else 0


if __name__ == "__main__":
    raise SystemExit(main())
