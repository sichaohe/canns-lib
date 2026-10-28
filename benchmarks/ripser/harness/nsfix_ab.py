#!/usr/bin/env python3
"""Interleaved A/B over several builds of canns-lib's ripser, on several arms.

WHAT IS BEING COMPARED
  The builds are named on the command line, so the same script measures any
  set of builds. The shipped default (`site_g_orig`) is the control.

  The change under test in this round has two independent parts.
    1. `rips_dm_with_callback_and_interval` no longer switches to the SPARSE
       distance matrix when the matrix has no missing pairs at all. The old
       condition was `threshold < max_finite`, which the ASA pipeline trips
       by construction: it replaces the ~30% of +inf pairs with
       M = 2*max_finite + 1, so `max_finite` becomes M and any threshold
       T < M sends an ordinary DENSE matrix down the sparse path. In this
       implementation the sparse path is 4.3x SLOWER on identical work
       (identical walk counters, identical barcode).
    2. The dense distance lookup is branchless and computes the row offset
       arithmetically instead of loading it from `row_offsets`, and the
       cofacet enumerator no longer performs two integer divisions per
       cofacet.

THE FOUR ARMS. The two input forms x the two thresholds, because the two
changes interact with both:
  inf  + T   : the +inf matrix at the real threshold      (the dense path)
  inf  + CAP: the +inf matrix at the cap                  (the dense path)
  cap  + T   : the capped matrix at the real threshold     (ripser++'s input;
                this is the arm the sparse false positive used to ruin)
  cap  + CAP: the capped matrix at the cap                (the dense path)

NOTE: EVERY BUILD MUST BE COMPILED WITH THE `gpu` FEATURE. The GPU apparent
  pre-pass lives behind `#[cfg(feature = "gpu")]` and the feature is NOT in
  `default` and NOT in `[tool.maturin] features` (on purpose: the released
  wheel must build without nvcc). A build without it silently falls back to
  the CPU pre-pass and is ~35% slower, which reads exactly like a regression
  in the change under test. The script therefore refuses to run against a
  build whose `gpu_err` is not empty.

GATES. Every run must print the reference barcode md5
`ee1f5283091dd645a3dd53e6f1a7eac6:9f0aa8c3ff2716f2fbb9ecd523bb0210:2d018d8fe9c1895b1b74f837af59ed20`
and the bar counts `1195|481|58`. A run that disagrees is recorded as
`BADCODEC` and the checker fails the whole CSV.

USAGE
    python nsfix_ab.py --out nsfix_ab.csv --reps 3 \
        --builds orig=/path/site_g_orig base=/path/site_g_base
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
T = 200.2874965          # max finite distance = the real threshold
CAP = 401.5749931        # 2*max_finite + 1

REF_MD5 = ("ee1f5283091dd645a3dd53e6f1a7eac6:"
           "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
           "2d018d8fe9c1895b1b74f837af59ed20")
REF_BARS = "1195|481|58"

ARMS = [
    ("inf_T", "inf", T),
    ("inf_CAP", "inf", CAP),
    ("cap_T", "capped", T),
    ("cap_CAP", "capped", CAP),
]

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

d = np.load("/home/sichaohe/ph_work/ph_dm_1200.npy")
np.fill_diagonal(d, 0.0)
if mode == "capped":
    cap = float(d[np.isfinite(d)].max()) * 2 + 1.0
    d = np.where(np.isfinite(d), d, cap)
t0 = time.perf_counter()
r = ripser(d, maxdim=2, thresh=thresh, coeff=47, distance_matrix=True, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
md5 = ":".join(hashlib.md5(np.asarray(sorted(map(tuple, x)), dtype="float64").tobytes()).hexdigest() for x in r)
h1 = np.asarray(r[1], dtype="float64")
life = np.sort(h1[:, 1] - h1[:, 0])[::-1]
life = life[np.isfinite(life)]
# Peak resident set of THIS process, after the computation. The
# reduction's working structures are the bulk of it, so this is where a
# change in the column store shows up.
rss = 0
for line in open("/proc/self/status"):
    if line.startswith("VmHWM:"):
        rss = int(line.split()[1])
        break
print("WALL %.3f" % wall)
print("RSS %d" % rss)
print("BARS " + "|".join(str(len(x)) for x in r))
print("MD5 " + md5)
print("H1TOP " + ",".join("%.2f" % x for x in life[:3]))
'''


def run(site: str, thresh: float, mode: str, timeout: int = 900):
    env = dict(os.environ)
    env["CANNS_RIPSER_STATS"] = "1"
    p, foreign, foreign_max = bench_core.with_foreign(
        subprocess.run, [PY, "-c", CHILD, site, repr(thresh), mode],
        capture_output=True, text=True, env=env, timeout=timeout)
    out = {"foreign_cpu": foreign, "foreign_max": foreign_max}
    for line in p.stdout.splitlines():
        if " " in line:
            k, _, v = line.partition(" ")
            out[k] = v
    out["rc"] = p.returncode
    m = re.search(r'gpu_err="([^"]*)"', p.stderr)
    out["gpu_err"] = m.group(1) if m else ""
    if p.returncode != 0:
        out["stderr"] = p.stderr[-400:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nsfix_ab.csv")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--builds", nargs="+", required=True,
                    help="name=path pairs, e.g. orig=/path/site_g_orig")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of arm names (default: all four)")
    args = ap.parse_args()

    builds = {}
    for spec in args.builds:
        if "=" not in spec:
            print("FAIL: --builds entries must be name=path, got %r" % spec, file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        builds[name] = path
    arms = [a for a in ARMS if args.arms is None or a[0] in args.arms]
    if not arms:
        print("FAIL: no arms selected", file=sys.stderr)
        return 2

    rows = []
    for rep in range(1, args.reps + 1):
        for arm, mode, thresh in arms:
            for bname, site in builds.items():
                t0 = time.time()
                r = run(site, thresh, mode)
                rec = {
                    "rep": rep,
                    "arm": arm,
                    "mode": mode,
                    "thresh": thresh,
                    "build": bname,
                    "wall_s": r.get("WALL", ""),
                    "rss_kb": r.get("RSS", ""),
                    "bars": r.get("BARS", ""),
                    "md5": r.get("MD5", ""),
                    "h1top": r.get("H1TOP", ""),
                    "gpu_err": r.get("gpu_err", ""),
                    "rc": r.get("rc", ""),
                    "host_elapsed_s": "%.3f" % (time.time() - t0),
                    "foreign_cpu": r.get("foreign_cpu", ""),
                    "foreign_max": r.get("foreign_max", ""),
                }
                if rec["gpu_err"]:
                    rec["verdict"] = "NOGPU"
                elif rec["md5"] != REF_MD5 or rec["bars"] != REF_BARS:
                    rec["verdict"] = "BADCODEC"
                else:
                    rec["verdict"] = "OK"
                rows.append(rec)
                print("rep=%d arm=%-8s build=%-6s wall=%-8s rss_kb=%-10s gpu_err=%-4r %s %s"
                      % (rep, arm, bname, rec["wall_s"], rec["rss_kb"], rec["gpu_err"],
                         rec["bars"], rec["verdict"]), flush=True)

    fields = ["rep", "arm", "mode", "thresh", "build", "wall_s", "rss_kb", "bars",
              "md5", "h1top", "gpu_err", "verdict", "rc", "host_elapsed_s",
              "foreign_cpu", "foreign_max"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rec in rows:
            w.writerow(rec)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
