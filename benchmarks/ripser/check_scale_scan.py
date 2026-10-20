#!/usr/bin/env python3
"""Checker for the scale scan.

  1. THE GRID IS COMPLETE. Every size has all three sides: our build on
     the `+inf` matrix, our build on the capped matrix, and ripser++ on
     the capped matrix. The denominator is computed from the size list.
  2. THE THREE SIDES COMPUTE THE SAME FILTRATION. Capping the missing
     pairs does not change WHICH simplices are in the filtration, only
     how many simplices have to be enumerated -- so our two input forms
     must produce the SAME barcode, and ripser++ must report the same
     number of bars. This is the invariant the whole experiment rests on.
  3. THE MEASUREMENT IS USABLE. A positive wall time, a peak RSS, the
     load average, and a GPU-memory reading for the ripser++ side -- the
     point of the scan is where the device runs out, so the free memory
     before the run has to be in the table.

USAGE
    python check_scale_scan.py scale_scan.csv [--sizes 1200,1600,2000] [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

SIDES = ("ours_inf", "ours_capped", "ripserpp")


def check(path, sizes=None):
    fails = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["table is empty: %s" % path]
    for col in ("n", "side", "wall_s", "rss_kb", "bars", "md5", "loadavg",
                "gpu_free_mib_before"):
        if col not in rows[0]:
            return ["table is missing column %s" % col]

    ns = sizes or sorted({int(r["n"]) for r in rows})
    print("expected cells: %d sizes x %d sides = %d"
          % (len(ns), len(SIDES), len(ns) * len(SIDES)))
    print("rows in table             : %d" % len(rows))

    by = {}
    for r in rows:
        by.setdefault(int(r["n"]), {})[r["side"]] = r
    for n in ns:
        for side in SIDES:
            if side not in by.get(n, {}):
                fails.append("missing cell: n=%d side=%s" % (n, side))
    for n in by:
        if n not in ns:
            fails.append("unexpected size: n=%d" % n)

    for n, sides in by.items():
        inf, cap, pp = sides.get("ours_inf"), sides.get("ours_capped"), \
            sides.get("ripserpp")
        if inf and cap:
            if inf["md5"] and cap["md5"] and inf["md5"] != cap["md5"]:
                fails.append("n=%d: the +inf and the capped input form give "
                             "different barcodes (%s against %s) -- the two "
                             "forms must be the same filtration"
                             % (n, inf["md5"], cap["md5"]))
        if inf and pp:
            if inf["bars"] and pp["bars"] and inf["bars"] != pp["bars"]:
                fails.append("n=%d: ripser++ reports %s bars, we report %s"
                             % (n, pp["bars"], inf["bars"]))
        for side, r in sides.items():
            try:
                if float(r["wall_s"]) <= 0:
                    fails.append("n=%d %s: non-positive wall time" % (n, side))
                if int(r["rss_kb"]) <= 0:
                    fails.append("n=%d %s: no peak RSS" % (n, side))
            except ValueError:
                fails.append("n=%d %s: the timing columns are not numbers"
                             % (n, side))
            if not (r.get("loadavg") or "").strip():
                fails.append("n=%d %s: no load average recorded" % (n, side))
        if pp and (pp.get("gpu_free_mib_before") or "").strip() in ("", "-1"):
            fails.append("n=%d ripserpp: no free GPU memory recorded -- the "
                         "point of the scan is where the device runs out"
                         % n)
    return fails


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    if kind == "drop_side":
        rows = [r for r in rows if not (r["side"] == "ours_capped"
                                        and r["n"] == rows[0]["n"])]
    elif kind == "differing_forms":
        for r in rows:
            if r["side"] == "ours_capped":
                r["md5"] = "0" * 16
                break
    elif kind == "bars_disagree":
        for r in rows:
            if r["side"] == "ripserpp":
                r["bars"] = "999999"
                break
    elif kind == "no_gpu_free":
        for r in rows:
            if r["side"] == "ripserpp":
                r["gpu_free_mib_before"] = "-1"
                break
    elif kind == "zero_time":
        rows[0]["wall_s"] = "0"
    elif kind == "no_loadavg":
        rows[0]["loadavg"] = ""
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["drop_side", "differing_forms", "bars_disagree", "no_gpu_free",
             "zero_time", "no_loadavg"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="scale_scan.csv")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()] or None

    fails = check(args.csv, sizes)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: every size has all three sides, the two input forms agree "
          "on the barcode, and the measurements are identified")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="scale_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, sizes)
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
