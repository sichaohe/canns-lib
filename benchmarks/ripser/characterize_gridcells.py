#!/usr/bin/env python3
"""Characterise the load the grid-cells corpus actually produces.

WHY THIS IS THE FIRST SECTION OF THE PAPER. Every performance number so
far comes from ONE matrix, and the "99.87% of the columns are apparent
pairs" figure is a figure about THAT matrix. A characterisation that is
only true of one instance is an anecdote. This script measures the
properties across the whole corpus, so the paper can state a DISTRIBUTION
(median, range) instead of a single number, and can say which properties
generalise and which are properties of the ASA matrix.

FOR EACH MATRIX IT RECORDS

  * the data: vertices, the fraction of point pairs that are genuinely
    missing (+inf), the edges in the filtration;
  * dim 1 and dim 2: the number of columns, how many of them are apparent
    pairs, and the wall time of each phase (GPU pre-pass, the reduction
    loop, the sharded pivot build);
  * the phase decomposition of the whole call.

The run is at maxdim=2, because the interesting regime (the one where the
apparent fast path decides the outcome) is dim 2. At maxdim=1 the corpus
never reaches the GPU path at all -- the reduction walks the EDGE columns,
719,400 of them at n=1200, which is below the threshold at which the GPU
pre-pass engages -- so a maxdim=1 characterisation would report the CPU
path and call it the corpus.

USAGE
    python characterize_gridcells.py --site <site> [--out characterize_gridcells.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time

import numpy as np

GRID = "/home/sichaohe/gridcells/matrices"
PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"

CHILD = r'''
import sys, time
import numpy as np
site, path, maxdim, coeff = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, site)
from canns_lib.ripser import ripser
d = np.load(path)
np.fill_diagonal(d, 0.0)
t0 = time.perf_counter()
r = ripser(d, maxdim=maxdim, thresh=float("inf"), coeff=coeff,
           distance_matrix=True, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
print("WALL %.4f" % wall)
print("BARS " + "|".join(str(len(x)) for x in r))
'''


def parse_stats(stderr):
    """The STATS / PHASESTATS lines into a dict."""
    out = {}
    for line in stderr.splitlines():
        m = re.match(r"STATS dim=(\d+) (.*)", line)
        if m:
            d = int(m.group(1))
            kv = {}
            for tok in m.group(2).split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kv[k] = v
            out["dim%d" % d] = kv
        m = re.match(r"PHASESTATS (.*)", line)
        if m:
            kv = {}
            for tok in m.group(1).split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kv[k] = v
            out["phase"] = kv
    return out


def num(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/home/sichaohe/canns-lib-dev/site_ship6")
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--coeff", type=int, default=47)
    ap.add_argument("--out", default="characterize_gridcells.csv")
    args = ap.parse_args()

    mats = sorted(f[:-4] for f in os.listdir(GRID) if f.endswith(".npy"))
    print("matrices: %d" % len(mats), flush=True)

    rows = []
    for i, m in enumerate(mats):
        path = os.path.join(GRID, m + ".npy")
        d = np.load(path)
        n = int(d.shape[0])
        fin = np.isfinite(d)
        npairs = n * (n - 1) // 2
        nfin = int((fin.sum() - n) // 2)          # the diagonal is always finite
        # the edges of the filtration: finite pairs, at any distance
        env = dict(os.environ)
        env["CANNS_RIPSER_STATS"] = "1"
        with open("/tmp/_char.out", "w") as fo:
            t0 = time.perf_counter()
            p = subprocess.run(["/usr/bin/time", "-f", "__T__%e %M", PY, "-c", CHILD,
                                args.site, path, str(args.maxdim), str(args.coeff)],
                               stdout=fo, stderr=subprocess.PIPE, text=True,
                               env=env, timeout=3600)
            wall = time.perf_counter() - t0
        st = parse_stats(p.stderr)
        d1 = st.get("dim1", {})
        d2 = st.get("dim2", {})
        ph = st.get("phase", {})
        m1 = re.search(r"__T__([\d.]+) (\d+)", p.stderr)
        row = {
            "matrix": m,
            "n": n,
            "pairs": npairs,
            "finite_pairs": nfin,
            "inf_frac": "%.4f" % (1.0 - nfin / npairs),
            "dim1_cols": d1.get("columns", ""),
            "dim1_apparent": d1.get("apparent_pairs", ""),
            "dim1_apparent_frac": ("%.6f" % (num(d1, "apparent_pairs") /
                                             num(d1, "columns"))) if d1.get("columns") else "",
            "dim1_loop_s": d1.get("loop_s", ""),
            "dim1_total_s": d1.get("total_s", ""),
            "dim1_gpu_pre_s": d1.get("gpu_pre_s", ""),
            "dim2_cols": d2.get("columns", ""),
            "dim2_apparent": d2.get("apparent_pairs", ""),
            "dim2_apparent_frac": ("%.6f" % (num(d2, "apparent_pairs") /
                                             num(d2, "columns"))) if d2.get("columns") else "",
            "dim2_loop_s": d2.get("loop_s", ""),
            "dim2_total_s": d2.get("total_s", ""),
            "dim2_gpu_pre_s": d2.get("gpu_pre_s", ""),
            "dim2_pivot_build_s": d2.get("sharded_pivot_build_s", ""),
            "ph_wall_s": ph.get("wall", ""),
            "ph_reduce_s": ph.get("reduce", ""),
            "ph_assemble_s": ph.get("assemble", ""),
            "wall_s": "%.4f" % wall,
            "rss_kb": m1.group(2) if m1 else "",
            "loadavg": open("/proc/loadavg").read().split()[0],
        }
        rows.append(row)
        print("[%d/%d] %-46s inf=%.3f d1_appr=%s d2_appr=%s wall=%.2fs"
              % (i + 1, len(mats), m, 1.0 - nfin / npairs,
                 row["dim1_apparent_frac"], row["dim2_apparent_frac"], wall),
              flush=True)

    fields = ["matrix", "n", "pairs", "finite_pairs", "inf_frac",
              "dim1_cols", "dim1_apparent", "dim1_apparent_frac",
              "dim1_loop_s", "dim1_total_s", "dim1_gpu_pre_s",
              "dim2_cols", "dim2_apparent", "dim2_apparent_frac",
              "dim2_loop_s", "dim2_total_s", "dim2_gpu_pre_s",
              "dim2_pivot_build_s",
              "ph_wall_s", "ph_reduce_s", "ph_assemble_s",
              "wall_s", "rss_kb", "loadavg"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
