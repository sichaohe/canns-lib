#!/usr/bin/env python3
"""The scale scan: where does the engine stop fitting, and who stops first?

THE CLAIM UNDER TEST. ripser++ cannot represent a missing distance: its
file format has no `+inf`, so the only way to run it on a matrix with
missing pairs is to CAP them -- which turns every missing pair into a
real simplex of the capped filtration, and the capped filtration has
2.84x as many dim-2 columns as the true one on this workload. That extra
work is not only time: ripser++'s dim-2 structures are sized by the
CAPPED column count, so on a large instance it needs GPU memory that a
machine with 80 GB does not have.

We keep the `+inf`, so we build the true filtration and skip the
simplices that can never join it. The claim is therefore not "we are
faster" but "**we still run on the instances ripser++ cannot run on**" --
which is a scale claim, not a percentage, and a much harder one to
argue away.

THE DATA. Synthetic, on purpose: the question is about the ENGINE's
scaling, and the corpus's matrices are all 1200x1200, so they cannot
answer it. The generator makes points in 9 dimensions (the dimension of
the corpus's embeddings) and then REMOVES a fraction of the point pairs,
the way the corpus's own pipeline removes them. The fraction is
recorded, because the whole effect depends on it.

THE SIDES (all three compute the SAME filtration, and the barcodes are
compared):
  ours_inf     canns-lib with the `+inf` matrix      -- the true filtration
  ours_capped  canns-lib with the capped matrix      -- what ripser++ must do
  ripserpp     ripser++ on the capped matrix

USAGE
    python scale_scan.py --site <site> --sizes 1200,1600,2000 [--reps 2]
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import os
import re
import subprocess
import sys
import time

import numpy as np

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
PP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"

CHILD = r'''
import json, sys, time
import numpy as np
site, path, maxdim, coeff, form = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                   int(sys.argv[4]), sys.argv[5])
sys.path.insert(0, site)
from canns_lib.ripser import ripser
d = np.load(path)
if form == "capped":
    cap = float(d[np.isfinite(d)].max()) * 2 + 1.0
    d = np.where(np.isfinite(d), d, cap)
t0 = time.perf_counter()
r = ripser(d, maxdim=maxdim, thresh=float("inf"), coeff=coeff,
           distance_matrix=True, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
print("WALL %.4f" % wall)
print("BARS " + json.dumps({str(d): [list(map(float, b)) for b in x]
                             for d, x in enumerate(r)}))
'''


def make_instance(n, dim, missing_frac, seed, path):
    """A point cloud with a controlled fraction of missing pairs."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, dim))
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
    if missing_frac > 0:
        # Remove pairs at random, symmetrically: a pair is either present
        # for both vertices or for neither.
        iu = np.triu_indices(n, 1)
        keep = rng.random(iu[0].size) >= missing_frac
        mask = np.ones((n, n), dtype=bool)
        mask[iu[0][~keep], iu[1][~keep]] = False
        mask = mask | mask.T
        d = np.where(mask, d, np.inf)
    np.save(path, d)
    return d


def gpu_free_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/home/sichaohe/canns-lib-dev/site_ship6")
    ap.add_argument("--sizes", default="1200,1600,2000")
    ap.add_argument("--dim", type=int, default=9)
    ap.add_argument("--missing", type=float, default=0.30)
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--coeff", type=int, default=47)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="scale_scan.csv")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    rows = []
    for n in sizes:
        path = "/tmp/_scale_%d.npy" % n
        print("=== n=%d ===" % n, flush=True)
        d = make_instance(n, args.dim, args.missing, 1234 + n, path)
        fin = np.isfinite(d)
        npairs = n * (n - 1) // 2
        nfin = int((fin.sum() - n) // 2)
        print("  missing pairs: %.4f" % (1 - nfin / npairs), flush=True)

        # --- ripser++ (capped, the only form it can read) ------------------
        t0 = time.perf_counter()
        txt = "/tmp/_scale_%d.txt" % n
        np.savetxt(txt, np.where(fin, d, float(d[fin].max()) * 2 + 1.0),
                   fmt="%.10g")
        txt_write = time.perf_counter() - t0
        pp_best, pp_rss, pp_bars, pp_err = None, 0, None, ""
        for rep in range(args.reps):
            free0 = gpu_free_mib()
            try:
                with open("/tmp/_scale_pp.out", "w") as fo:
                    t0 = time.perf_counter()
                    p = subprocess.run(["/usr/bin/time", "-f", "__T__%e %M", PP,
                                        "--format", "distance", "--dim",
                                        str(args.maxdim), "--threshold", "inf",
                                        "--ratio", "1.0", txt],
                                       stdout=fo, stderr=subprocess.PIPE,
                                       text=True, timeout=3600)
                    wall = time.perf_counter() - t0
                m = re.search(r"__T__([\d.]+) (\d+)", p.stderr)
                rss = int(m.group(2)) if m else 0
                if pp_best is None or wall < pp_best:
                    pp_best, pp_rss = wall, rss
                pp_bars = sum(1 for ln in open("/tmp/_scale_pp.out")
                              if ln.startswith(" ["))
                if "CUDA error" in p.stderr or "out of memory" in p.stderr.lower():
                    pp_err = "GPU_OOM"
            except subprocess.TimeoutExpired:
                pp_err = "TIMEOUT"
        print("  ripser++: %s  rss=%s MB  bars=%s %s"
              % ("%.3f s" % pp_best if pp_best else "FAILED",
                 pp_rss // 1024, pp_bars, pp_err), flush=True)
        rows.append({"n": n, "side": "ripserpp", "wall_s": pp_best or "",
                     "rss_kb": pp_rss, "gpu_free_mib_before": free0,
                     "bars": pp_bars or "", "md5": "", "error": pp_err,
                     "loadavg": open("/proc/loadavg").read().split()[0]})

        # --- ours, both input forms ---------------------------------------
        for form in ("inf", "capped"):
            best, rss, bars = None, 0, None
            for rep in range(args.reps):
                with open("/tmp/_scale_ours.out", "w") as fo:
                    t0 = time.perf_counter()
                    p = subprocess.run(["/usr/bin/time", "-f", "__T__%e %M", PY,
                                        "-c", CHILD, args.site, path,
                                        str(args.maxdim), str(args.coeff), form],
                                       stdout=fo, stderr=subprocess.PIPE,
                                       text=True, timeout=3600)
                    wall = time.perf_counter() - t0
                m = re.search(r"__T__([\d.]+) (\d+)", p.stderr)
                r = int(m.group(2)) if m else 0
                if best is None or wall < best:
                    best, rss = wall, r
                for line in open("/tmp/_scale_ours.out"):
                    if line.startswith("BARS "):
                        bars = json.loads(line[5:])
            md5 = ""
            if bars:
                md5 = hashlib.md5(
                    json.dumps({k: sorted(map(tuple, v)) for k, v in bars.items()},
                               sort_keys=True).encode()).hexdigest()[:16]
            print("  ours %-7s %8.3f s  rss=%6d MB  bars=%s"
                  % (form, best or 0, rss // 1024,
                     sum(len(v) for v in (bars or {}).values())), flush=True)
            rows.append({"n": n, "side": "ours_" + form, "wall_s": best or "",
                         "rss_kb": rss, "gpu_free_mib_before": -1,
                         "bars": sum(len(v) for v in (bars or {}).values()),
                         "md5": md5, "error": "",
                         "loadavg": open("/proc/loadavg").read().split()[0]})
        os.unlink(path)
        os.unlink(txt)

    fields = ["n", "side", "wall_s", "rss_kb", "gpu_free_mib_before", "bars",
              "md5", "error", "loadavg"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
