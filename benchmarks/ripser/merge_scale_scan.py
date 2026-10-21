#!/usr/bin/env python3
"""Merge the passes of the scale scan into one table.

WHY THERE ARE PASSES. The reference's arms at n=2000 do not finish -- they
have to be timed out, and a timeout costs the whole limit (200 s each), so
running them together with our arms is expensive. Our arms at n=2000 are
also the ones that need a clean GPU: they allocate tens of gigabytes of
device memory, and a killed predecessor leaves the driver holding it, which
makes the next run slow enough to time out as well.

So the scan was run in passes, and this merges them. Every row records the
pass it came from, and the rows are the same measurement (the same
instance, the same build, the same arms) -- only the wall clock of the
passes differs, and the pass that produced a row is in the table.

USAGE
    python merge_scale_scan.py --out scale_scan.csv pass1.csv pass2.csv ...
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

FIELDS = ["n", "side", "threshold", "wall_s", "lib_wall_s", "rss_kb", "bars",
          "bars_d0", "bars_d1", "bars_d2", "error", "note", "loadavg",
          "threads", "commit", "build_md5", "pass"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default="scale_scan.csv")
    ap.add_argument("--stderr", default="",
                    help="the stderr of a reference run; its own words about "
                         "the device memory it needed are copied into the "
                         "note of the timed-out rows")
    args = ap.parse_args()
    # ripser++'s own words about the device memory it wanted, and about the
    # fallback it took. Copied verbatim from its stderr -- a timeout is only
    # a RESULT if it says WHY.
    from_stderr = ""
    if args.stderr and os.path.exists(args.stderr):
        txt = open(args.stderr, errors="replace").read()
        bits = []
        m = re.search(r"gpu memory needed for full rips by calculation in "
                      r"bytes for dim: \d+: (\d+)", txt)
        if m:
            bits.append("needs %.1f GB of device memory"
                        % (int(m.group(1)) / 1e9))
        if "CPU-ONLY MODE FOR REMAINDER" in txt:
            bits.append("FELL BACK TO CPU-ONLY (not enough device memory)")
        m = re.search(r"recalculated dim_max based on GPU free DRAM capacity: "
                      r"(\d+)", txt)
        if m:
            bits.append("recalculated dim_max to %s" % m.group(1))
        from_stderr = "; ".join(bits)

    rows = []
    for i, path in enumerate(args.csvs, 1):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                row = {k: r.get(k, "") for k in FIELDS}
                row["pass"] = "p%d" % i
                if (row.get("error") or "").strip() == "TIMEOUT" and from_stderr:
                    # do not repeat what the arm already said about itself
                    if from_stderr not in (row.get("note") or ""):
                        row["note"] = (row.get("note", "") + ";"
                                       if row.get("note") else "") + from_stderr
                rows.append(row)
    # the later pass wins for the same (n, side): it is the one that ran on
    # a clean machine
    best = {}
    for r in rows:
        best[(r["n"], r["side"])] = r
    order = sorted(best.values(), key=lambda r: (int(r["n"]), r["side"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in order:
            w.writerow(r)
    print("wrote %s (%d rows)" % (args.out, len(order)))
    for r in order:
        print("  n=%-5s %-14s %-6s %9s s  rss=%8s MB  bars=%-6s %s"
              % (r["n"], r["side"], r["threshold"], r["wall_s"],
                 int(r["rss_kb"]) // 1024 if r["rss_kb"] else "", r["bars"],
                 r["error"] or r["note"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
