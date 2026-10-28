#!/usr/bin/env python3
"""The unified ablation table: all switches on, against exactly one off.

THE FRAMING, and why it is this one. The paper's central table is an
ablation. There are two ways to build one, and they are NOT the same
measurement:

  "one switch ON at a time" -- build a variant that has only that change.
  "all switches ON, one OFF" -- build a variant that is the shipping build
  with exactly that change reverted.

The second is the right one here, because the changes were not introduced
as independent commits: the auto-sparse fix and the heap change arrived in
the same commit, and the column-store rewrite and the host/device hand-off
touch the same files. A "before" build from that history therefore differs
from the shipping build by MORE than the change being measured, and the
numbers would not be comparable with each other.

So this table measures the switches that CAN be turned off independently,
and it measures them against the shipping build:

  all_on              the shipping build, with every switch on
  heap_off            the same source, built with the `binary-heap`
                      feature: the reduction's working heap is
                      `std::collections::BinaryHeap` again, whose `pop`
                      walks the hole to the bottom of the tree
  auto_sparse_off     the same source, built with the `legacy-auto-sparse`
                      feature: the sparse representation is chosen
                      whenever the threshold is below the maximum finite
                      distance, without requiring the matrix to actually
                      contain a missing pair

Both are pure build flags, so the two variants differ from the shipping
build by exactly one thing, and the rows are comparable with each other.

The other two changes -- the column store's one boundary pair per
non-empty column, and the host/device hand-off -- are NOT independently
switchable in the current tree, and they are measured in their own tables
(`bounds_ab.csv`, `gpucopy_ab.csv`) as cumulative before/after pairs from
an earlier base commit. The paper's ablation section has two tables with
two headers for exactly this reason: the numbers are comparable within a
row and NOT across the two tables.

THE ARMS. The two input forms x the two thresholds, because the changes
interact with both.

GATES. Every run must print the reference barcode md5 and the bar counts.
A run that disagrees is recorded as `BADCODEC` and the checker fails the
whole table: an ablation row that changed the answer is not an ablation
row, it is a bug.

USAGE
    python unified_ablation.py --out unified_ablation.csv --reps 3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import time

# How much of the box was used by somebody ELSE while each variant ran --
# see foreign_cpu.py for why the load average cannot answer it.
import foreign_cpu
# The measurement lock lives in bench_core, and this import has to be in the
# PARENT's scope. The one inside the child script string further down is the
# child's: the parent never gets it, so `bench_core.measurement_lock` raised
# NameError and the run died before it measured anything.
import bench_core

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

# The shipping build, and the two variants that differ from it by exactly
# one cargo feature. The variants are built by
# `benchmarks/ripser/drivers/build_abl_variants.sh`.
VARIANTS = [
    ("all_on", "shipping", "/home/sichaohe/canns-lib-dev/site_ship6"),
    ("heap_off", "binary-heap", "/home/sichaohe/canns-lib-dev/site_abl_heapoff"),
    ("auto_sparse_off", "legacy-auto-sparse",
     "/home/sichaohe/canns-lib-dev/site_abl_autosparseoff"),
]

FIELDS = ["rep", "variant", "feature", "arm", "mode", "thresh", "build",
          "wall_s", "rss_kb", "bars", "md5", "h1top", "gpu_err", "rc",
          "verdict", "loadavg", "foreign_cpu", "foreign_max", "gpu",
          "commit"]

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


def source_commit():
    src = "/home/sichaohe/canns-lib-dev"
    try:
        c = subprocess.run(["git", "-C", src, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        if c:
            return c
    except Exception:
        pass
    h = hashlib.md5()
    for rel in ("src/ripser/core/reduction.rs", "src/ripser/matrix/sparse.rs",
                "src/ripser/core/algorithm.rs", "Cargo.toml"):
        try:
            h.update(open(os.path.join(src, rel), "rb").read())
        except OSError:
            pass
    return "src:%s" % h.hexdigest()[:12]


def run(site: str, thresh: float, mode: str, timeout: int = 900, env=None):
    env = dict(env or os.environ)
    env["CANNS_RIPSER_STATS"] = "1"
    p = subprocess.run([PY, "-c", CHILD, site, repr(thresh), mode],
                       capture_output=True, text=True, env=env, timeout=timeout)
    out = {}
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


def write_table(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="unified_ablation.csv")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    arms = [a for a in ARMS if args.arms is None or a[0] in args.arms]
    variants = [v for v in VARIANTS
                if args.variants is None or v[0] in args.variants]
    if not arms or not variants:
        print("FAIL: nothing selected", file=sys.stderr)
        return 2

    dev = os.environ.get("CANNS_BENCH_GPU", "0")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev)
    commit = source_commit()
    print("device=%s  commit=%s  reps=%d  variants=%s"
          % (dev, commit, args.reps, [v[0] for v in variants]), flush=True)
    # THE MEASUREMENT LOCK, taken for the whole run. These runners spawn their
    # children directly rather than through bench_core, so the lock has to be
    # taken here: the quiet-window gate counts the cores used by processes
    # that are NOT ours, and it cannot see two of our own measurements
    # running beside each other -- which moves the numbers exactly as much as
    # a neighbour's job does. The lock is released when the process exits.
    _LOCK = bench_core.measurement_lock(label="unified_ablation")
    _LOCK.__enter__()

    rows = []
    for rep in range(1, args.reps + 1):
        for arm, mode, thresh in arms:
            # the shipping build and each variant, back to back, so the
            # machine state is as close to the same for the two as it can be
            for name, feature, path in variants:
                if not os.path.isdir(path):
                    print("FAIL: the variant's site is missing: %s" % path,
                          file=sys.stderr)
                    return 2
                t0 = time.time()
                # How much of the box was used by somebody ELSE while this
                # variant ran -- see foreign_cpu.py for why the load average
                # cannot answer it.
                probe = foreign_cpu.Probe()
                try:
                    with probe:
                        r = run(path, thresh, mode, args.timeout, env=env)
                except subprocess.TimeoutExpired as e:
                    try:
                        e.foreign = probe.record()
                        e.foreign_max = probe.record_max()
                    except Exception:
                        pass
                    raise
                foreign = probe.record()
                foreign_max = probe.record_max()
                rec = {
                    "rep": rep, "variant": name, "feature": feature,
                    "arm": arm, "mode": mode, "thresh": thresh,
                    "build": os.path.basename(path),
                    "wall_s": r.get("WALL", ""),
                    "rss_kb": r.get("RSS", ""),
                    "bars": r.get("BARS", ""),
                    "md5": r.get("MD5", ""),
                    "h1top": r.get("H1TOP", ""),
                    "gpu_err": r.get("gpu_err", ""),
                    "rc": r.get("rc", ""),
                    "loadavg": open("/proc/loadavg").read().split()[0],
                    "foreign_cpu": foreign,
                    "foreign_max": foreign_max,
                    "gpu": dev, "commit": commit,
                }
                if rec["gpu_err"]:
                    rec["verdict"] = "NOGPU"
                elif rec["md5"] != REF_MD5 or rec["bars"] != REF_BARS:
                    rec["verdict"] = "BADCODEC"
                else:
                    rec["verdict"] = "OK"
                rows.append(rec)
                print("rep=%d %-16s %-8s wall=%-8s rss=%-9s %s %s"
                      % (rep, name, arm, rec["wall_s"], rec["rss_kb"],
                         rec["bars"], rec["verdict"]), flush=True)
        write_table(args.out, rows)

    write_table(args.out, rows)
    bad = sorted({(r["variant"], r["arm"]) for r in rows
                  if r["verdict"] != "OK"})
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    print("runs that did not reproduce the reference barcode: %d" % len(bad))
    for c in bad:
        print("   %s / %s" % c)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
