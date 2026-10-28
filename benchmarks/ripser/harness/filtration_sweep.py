#!/usr/bin/env python3
"""Both filtrations, both sides, one interleaved batch.

The head-to-head was repeatedly wrong because the two sides were fed
different filtrations. This script makes that impossible: it runs FOUR arms
of canns-lib (two input forms x two thresholds) and TWO arms of ripser++
(the two thresholds, on the full-matrix file), all in ONE interleaved batch
on one machine, and writes them to a single CSV with the filtration
explicit in every row.

THE FOUR ARMS
    inf_T    : the +inf matrix, threshold T = max finite distance
               (100,897,898 dim-2 columns) -- the canonical ASA filtration
    inf_CAP  : the +inf matrix, threshold CAP  (286,562,199 dim-2 columns)
    cap_T    : the capped matrix, threshold T  (100,897,898 dim-2 columns)
    cap_CAP  : the capped matrix, threshold CAP (286,562,199 dim-2 columns)

THE TWO ARMS OF ripser++
    T        : the full-matrix file, --threshold T
    CAP      : the full-matrix file, --threshold CAP

capped+T and inf+T compute the SAME filtration (the finite part of the
matrix), and so do inf_CAP and cap_CAP. The checker asserts that the
paired arms agree on the bar count, so a mis-paired comparison fails
instead of producing a plausible-looking ratio.

USAGE
    python filtration_sweep.py --out filtration_sweep.csv [--reps 2]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
ASA_NPY = "/home/sichaohe/ph_work/ph_dm_1200.npy"
PP_TXT = "/tmp/asa1200_full.txt"
RP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"
SITE = os.environ.get("CANNS_RIPSER_SITE", "/home/sichaohe/canns-lib-dev/site_ship2")

T = 200.2874965          # max finite distance
CAP = 401.5749931        # 2*max_finite + 1

# arm -> (input form, threshold, the ripser++ arm it must agree with)
ARMS = [
    ("inf_T", "inf", T, "T"),
    ("inf_CAP", "inf", CAP, "CAP"),
    ("cap_T", "capped", T, "T"),
    ("cap_CAP", "capped", CAP, "CAP"),
]
PP_ARMS = [("T", T), ("CAP", CAP)]

REF_MD5 = ("ee1f5283091dd645a3dd53e6f1a7eac6:"
           "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
           "2d018d8fe9c1895b1b74f837af59ed20")
REF_BARS = "1195|481|58"

# The shared core: the host's paths, the foreign-load-measured child runner,
# the parsers, the CSV writer and the build identity. This import has to be
# HERE, in the parent's scope: the one inside the child script string below
# is the child's, so `bench_core.with_foreign` raised NameError and the runner
# died before it measured anything.
import bench_core

CHILD = r'''
import hashlib, sys, time
import numpy as np
site, thresh, mode = sys.argv[1], float(sys.argv[2]), sys.argv[3]
sys.path.insert(0, site)
from canns_lib.ripser import ripser

import bench_core
d = np.load("/home/sichaohe/ph_work/ph_dm_1200.npy")
np.fill_diagonal(d, 0.0)
if mode == "capped":
    cap = float(d[np.isfinite(d)].max()) * 2 + 1.0
    d = np.where(np.isfinite(d), d, cap)
# WARM-UP: the first call in a process pays the one-time CUDA context
# creation (0.32 s), which ripser++'s internal timer excludes. See the
# comment in clean_h2h.py.
t0 = time.perf_counter()
ripser(d, maxdim=2, thresh=thresh, coeff=47, distance_matrix=True, progress_bar=False)["dgms"]
cold = time.perf_counter() - t0
t0 = time.perf_counter()
r = ripser(d, maxdim=2, thresh=thresh, coeff=47, distance_matrix=True, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
md5 = ":".join(hashlib.md5(np.asarray(sorted(map(tuple, x)), dtype="float64").tobytes()).hexdigest() for x in r)
print("WALL %.3f" % wall)
print("WALL_COLD %.3f" % cold)
print("BARS " + "|".join(str(len(x)) for x in r))
print("MD5 " + md5)
'''

FIELDS = ["rep", "side", "build", "arm", "filtration", "input_form", "thresh",
          "wall_s", "wall_cold_s", "proc_s", "bars", "md5", "verdict",
          "foreign_cpu", "foreign_max"]


def run_ours(mode, thresh, timeout=900):
    p, foreign, foreign_max = bench_core.with_foreign(
        subprocess.run,
        ["/usr/bin/time", "-f", "__T__%e", PY, "-c", CHILD,
         SITE, repr(thresh), mode],
        capture_output=True, text=True, timeout=timeout)
    out = {"foreign_cpu": foreign, "foreign_max": foreign_max}
    for line in p.stdout.splitlines():
        if " " in line:
            k, _, v = line.partition(" ")
            out[k] = v
    m = re.search(r"__T__([\d.]+)", p.stderr)
    out["PROC"] = m.group(1) if m else ""
    out["rc"] = p.returncode
    return out


def run_pp(thresh, timeout=900):
    p, foreign, foreign_max = bench_core.with_foreign(
        subprocess.run,
        ["/usr/bin/time", "-f", "__T__%e", RP, "--format", "distance",
         "--dim", "2", "--threshold", "%.10g" % thresh, "--ratio", "1.0", PP_TXT],
        capture_output=True, text=True, timeout=timeout)
    m = re.search(r"__T__([\d.]+)", p.stderr)
    # `wall_s` is the LIBRARY wall on both sides: ripser++'s own "total
    # time", which excludes its process start, its CUDA init and the load of
    # the text file. The process wall is kept in `proc_s` so both
    # conventions are in the table, and the head-to-head table quotes the
    # same one. Quoting the process wall here while the head-to-head quoted
    # the library wall made two tables of the same comparison disagree by
    # 5%.
    lib = re.search(r"total time: ([\d.]+)s", p.stderr)
    bars = sum(1 for ln in p.stdout.splitlines() if ln.startswith(" ["))
    return {"wall_s": lib.group(1) if lib else "",
            "proc_s": m.group(1) if m else "",
            "bars": str(bars), "rc": p.returncode,
            "foreign_cpu": foreign, "foreign_max": foreign_max}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="filtration_sweep.csv")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    rows = []
    for rep in range(1, args.reps + 1):
        for arm, mode, thresh, pp_arm in ARMS:
            r = run_ours(mode, thresh)
            verdict = "OK" if (r.get("MD5") == REF_MD5 and r.get("BARS") == REF_BARS) else "BADCODEC"
            rows.append({"rep": rep, "side": "ours", "build": SITE,
                         "arm": arm, "filtration": pp_arm,
                         "input_form": mode, "thresh": thresh,
                         "wall_s": r.get("WALL", ""), "proc_s": r.get("PROC", ""),
                         "wall_cold_s": r.get("WALL_COLD", ""),
                         "bars": r.get("BARS", ""),
                         "md5": r.get("MD5", ""), "verdict": verdict,
                         "foreign_cpu": r.get("foreign_cpu", ""),
                         "foreign_max": r.get("foreign_max", "")})
            print("rep=%d ours %-8s %.3f %s" % (rep, arm, float(r.get("WALL", 0)), verdict), flush=True)
        for pp_arm, thresh in PP_ARMS:
            r = run_pp(thresh)
            rows.append({"rep": rep, "side": "ripserpp", "build": "ripser++ build3",
                         "arm": pp_arm, "filtration": pp_arm,
                         "input_form": "capped", "thresh": thresh,
                         "wall_s": r.get("wall_s", ""), "proc_s": r.get("proc_s", ""),
                         "bars": r.get("bars", ""),
                         "md5": "", "verdict": "OK" if r.get("bars") == "1734" else "BADCODEC",
                         "foreign_cpu": r.get("foreign_cpu", ""),
                         "foreign_max": r.get("foreign_max", "")})
            print("rep=%d ripserpp %-4s %s %s" % (rep, pp_arm, r.get("wall_s"), r.get("bars")), flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
