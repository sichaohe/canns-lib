#!/usr/bin/env python3
"""Is the reference implementation reproducible run to run?

MOTIVATION. Two runs of ripser++ on the SAME instance gave 10,988 and
10,946 bars, while our implementation gave 10,946 both times. The
difference is 42 bars out of 10,946 (0.4%), and it is in the H2 classes:
the GPU reduction's order is not fixed, so a run of the algorithm can
resolve a near-tie differently.

That matters for the paper in two ways. First, a bit-exact comparison
against ripser++ is not available on this workload, which is why every
barcode comparison in this work is a tolerance comparison. Second, our
own reproducibility is a property worth stating.

This script runs each side N times on ONE instance and reports the
distinct bar counts (and, for us, the distinct barcode md5s).

USAGE
    python determinism_check.py --site <site> [--n 1200] [--reps 3]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
PP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"

CHILD = r'''
import hashlib, json, sys, time
import numpy as np
site, path, coeff = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, site)
from canns_lib.ripser import ripser
d = np.load(path)
t0 = time.perf_counter()
r = ripser(d, maxdim=2, thresh=float("inf"), coeff=coeff,
           distance_matrix=True, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
bars = {d: [list(map(float, b)) for b in x] for d, x in enumerate(r)}
md5 = hashlib.md5(json.dumps({str(k): sorted(map(tuple, v))
                              for k, v in bars.items()},
                             sort_keys=True).encode()).hexdigest()[:16]
print("WALL %.3f" % wall)
print("BARS " + json.dumps(bars))
print("MD5 %s" % md5)
'''


def make_instance(n, dim=9, missing_frac=0.30, seed=1234):
    rng = np.random.default_rng(seed + n)
    pts = rng.normal(size=(n, dim))
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
    iu = np.triu_indices(n, 1)
    keep = rng.random(iu[0].size) >= missing_frac
    mask = np.ones((n, n), dtype=bool)
    mask[iu[0][~keep], iu[1][~keep]] = False
    mask = mask & mask.T
    return np.where(mask, d, np.inf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/home/sichaohe/canns-lib-dev/site_ship6")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--coeff", type=int, default=47)
    ap.add_argument("--out", default="determinism_check.csv")
    args = ap.parse_args()

    path = "/tmp/_det_%d.npy" % args.n
    d = make_instance(args.n)
    np.save(path, d)
    txt = "/tmp/_det_%d.txt" % args.n
    # ripser++'s reader cannot express `+inf` -- a text file with `inf` in it
    # is read back as a different matrix (it reported 648 bars instead of
    # 10,946 when the file was written with the holes intact). The CAPPED
    # matrix is the same filtration, so it is the right thing to hand it.
    cap = float(d[np.isfinite(d)].max()) * 2 + 1.0
    np.savetxt(txt, np.where(np.isfinite(d), d, cap), fmt="%.10g")

    rows = []
    for side in ("ours", "ripserpp"):
        counts, md5s, walls = [], [], []
        for rep in range(args.reps):
            if side == "ours":
                p = subprocess.run([PY, "-c", CHILD, args.site, path,
                                    str(args.coeff)],
                                   capture_output=True, text=True, timeout=3600)
                out = p.stdout
            else:
                with open("/tmp/_det_pp.out", "w") as fo:
                    t0 = time.perf_counter()
                    subprocess.run([PP, "--format", "distance", "--dim", "2",
                                    "--threshold", "inf", "--ratio", "1.0", txt],
                                   stdout=fo, stderr=subprocess.DEVNULL,
                                   timeout=3600)
                    walls.append(time.perf_counter() - t0)
                n = 0
                for ln in open("/tmp/_det_pp.out"):
                    if ln.startswith(" ["):
                        n += 1
                counts.append(n)
                md5s.append("n/a")
                print("  %-9s rep%d  bars=%d  wall=%.1f s"
                      % (side, rep + 1, n, walls[-1]), flush=True)
                continue
            bars, md5, wall = None, None, None
            for line in out.splitlines():
                if line.startswith("BARS "):
                    bars = json.loads(line[5:])
                elif line.startswith("MD5 "):
                    md5 = line.split()[1]
                elif line.startswith("WALL "):
                    wall = float(line.split()[1])
            n = sum(len(v) for v in bars.values()) if bars else 0
            counts.append(n)
            md5s.append(md5)
            walls.append(wall or 0.0)
            print("  %-9s rep%d  bars=%d  md5=%s  wall=%.1f s"
                  % (side, rep + 1, n, md5, wall or 0.0), flush=True)
        rows.append({"side": side, "n": args.n, "reps": args.reps,
                     "bars_distinct": "|".join(str(c) for c in sorted(set(counts))),
                     "md5_distinct": "|".join(sorted(set(m for m in md5s if m))),
                     "wall_min_s": "%.1f" % min(walls),
                     "wall_max_s": "%.1f" % max(walls)})
        print("%-9s: %d distinct bar count(s) %s, %d distinct md5 %s"
              % (side, len(set(counts)), sorted(set(counts)),
                 len(set(m for m in md5s if m)), sorted(set(m for m in md5s if m))),
              flush=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
