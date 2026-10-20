#!/usr/bin/env python3
"""A FAIR CPU baseline: giotto-ph's Ripser, which is multi-process.

WHY. The reference CPU implementation in the standard suite is the `ripser`
C++ binary, and the classic Ripser is SINGLE-THREADED, while our CPU build
uses every core. Quoting "7-10x faster than the reference CPU
implementation" without saying that is not a fair comparison, and the
review is right to flag it.

giotto-ph's Ripser is the same C++ reduction with a multiprocessing pool
over the columns, so it is the multi-process baseline our CPU build has to
beat. It is only feasible on the SMALL instances: at maxdim=2 a 1200-point
input needs C(1200,3) = 2.87e8 simplices, which does not fit.

The table therefore covers the datasets of the standard suite that fit,
and the comparison is on the SAME filtration (the capped matrix, the form
every implementation can read) with the barcodes compared.

USAGE
    python cpu_baseline_giotto.py [--site <site>] [--out cpu_giotto.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time

import numpy as np

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
EXAMPLES = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/examples"
PP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"

# (key, file, maxdim, threshold) -- only the instances that fit, and at
# maxdim=2: giotto-ph's Ripser builds the whole k-skeleton, so maxdim=3 on
# 300 points means 330M tetrahedra. The dimension is LOWER than the suite's
# on purpose, and the table records it.
DATASETS = [
    ("sphere_3_192", "sphere_3_192.distance_matrix.lower_triangular", 2, None),
    ("Vicsek300", "Vicsek300_300_of_300.distance_matrix", 2, None),
    ("celegans", "celegans.distance_matrix", 2, None),
]

CHILD = r'''
import json, sys, time
import numpy as np
site, path, maxdim, coeff = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, site)
from canns_lib.ripser import ripser
d = np.load(path)
t0 = time.perf_counter()
r = ripser(d, maxdim=maxdim, thresh=float("inf"), coeff=coeff,
           distance_matrix=True, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
bars = {d: [list(map(float, b)) for b in x] for d, x in enumerate(r)}
print("WALL %.4f" % wall)
print("BARS " + json.dumps(bars))
'''


def load(path):
    if path.endswith("lower_triangular"):
        rows = []
        for line in open(path):
            line = line.strip().rstrip(",")
            rows.append([float(x) for x in line.split(",") if x.strip()])
        n = len(rows)
        m = np.zeros((n, n), dtype=np.float64)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                if j < i:
                    m[i, j] = v
                    m[j, i] = v
        return m
    return np.loadtxt(path, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/home/sichaohe/canns-lib-dev/site_ship6")
    ap.add_argument("--coeff", type=int, default=2)
    ap.add_argument("--out", default="cpu_giotto.csv")
    args = ap.parse_args()

    rows = []
    for key, fname, maxdim, _th in DATASETS:
        path = os.path.join(EXAMPLES, fname)
        if not os.path.exists(path):
            print("%-14s missing" % key)
            continue
        d = load(path)
        cap = float(d[np.isfinite(d)].max()) * 2 + 1.0
        dc = np.where(np.isfinite(d), d, cap)
        npy = "/tmp/_giotto_%s.npy" % key
        np.save(npy, dc)

        # --- ours, on the CPU build ------------------------------------
        p = subprocess.run([PY, "-c", CHILD, args.site, npy, str(maxdim),
                            str(args.coeff)], capture_output=True, text=True,
                           timeout=7200)
        ours_wall = ours_bars = None
        for line in p.stdout.splitlines():
            if line.startswith("WALL "):
                ours_wall = float(line.split()[1])
            elif line.startswith("BARS "):
                # the TOTAL number of bars, not the number of dimensions
                ours_bars = sum(len(v) for v in json.loads(line[5:]).values())
        print("%-14s ours(cpu) %8.2f s  bars=%s" % (key, ours_wall or 0, ours_bars),
              flush=True)

        # --- giotto-ph, multi-process -----------------------------------
        gi_wall = gi_bars = None
        gi_err = ""
        try:
            code = ("import time, numpy as np, sys\n"
                    "from giotto.ph import Ripser\n"
                    "X = np.load(%r)\n"
                    "t0 = time.perf_counter()\n"
                    "R = Ripser(maxdim=%d, n_jobs=-1)\n"
                    "R.fit_transform(X)\n"
                    "print('WALL', time.perf_counter() - t0)\n"
                    % (npy, maxdim))
            q = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                               timeout=7200)
            for line in q.stdout.splitlines():
                if line.startswith("WALL "):
                    gi_wall = float(line.split()[1])
            if q.returncode != 0:
                gi_err = (q.stderr or "").strip().splitlines()[-1][:120] \
                    if q.stderr else "failed"
        except subprocess.TimeoutExpired:
            gi_err = "TIMEOUT"
        print("%-14s giotto    %8.2f s  %s" % (key, gi_wall or 0, gi_err), flush=True)

        # --- the reference C++ binary, single-threaded ------------------
        txt = "/tmp/_giotto_%s.txt" % key
        np.savetxt(txt, dc, fmt="%.10g")
        with open("/tmp/_giotto_pp.out", "w") as fo:
            t0 = time.perf_counter()
            subprocess.run([PP, "--format", "distance", "--dim", str(maxdim),
                            "--threshold", "inf", "--ratio", "1.0", txt],
                           stdout=fo, stderr=subprocess.DEVNULL, timeout=7200)
            pp_wall = time.perf_counter() - t0
        pp_bars = sum(1 for ln in open("/tmp/_giotto_pp.out") if ln.startswith(" ["))
        print("%-14s ripser++  %8.2f s  bars=%d" % (key, pp_wall, pp_bars),
              flush=True)

        rows.append({"dataset": key, "maxdim": maxdim,
                     "ours_threads": len(os.sched_getaffinity(0)),
                     "giotto_threads": "n/a",
                     "ours_cpu_s": "%.2f" % (ours_wall or 0),
                     "giotto_s": "%.2f" % (gi_wall or 0),
                     "giotto_error": gi_err,
                     "ripserpp_gpu_s": "%.2f" % pp_wall,
                     "ours_bars": ours_bars, "ripserpp_bars": pp_bars,
                     "loadavg": open("/proc/loadavg").read().split()[0]})
        os.unlink(npy)
        os.unlink(txt)

    fields = ["dataset", "maxdim", "ours_threads", "giotto_threads",
              "ours_cpu_s", "giotto_s", "giotto_error",
              "ripserpp_gpu_s", "ours_bars", "ripserpp_bars", "loadavg"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
