#!/usr/bin/env python3
"""The coefficient field, settled: is the ASA head-to-head the same
computation on both sides?

The classical-suite table is Z/2 on both sides, because ripser++ only does
Z/2. But the tables built on the ASA matrix -- the head-to-head, the
filtration sweep, the scale scan -- run OUR side over **Z/47**, because
that dataset is usually quoted with a large prime. Two consequences:

* the barcodes agree (the ASA matrix has no torsion in dimensions 0-2:
  1195 + 481 + 58 = 1734 bars, which is exactly what ripser++ reports),
  so the two sides compute the same homology;
* but "the same homology" is not the same as "the same computation", and
  the comparison should not rest on that.

This script answers it by running the three arms **interleaved in one
window**, with the order rotating on every repetition so that a drift in
the machine's own load cancels out of the comparison:

    ours over Z/47   ours over Z/2   ripser++ over Z/2

If our two arms land within a couple of percent of each other, the
coefficient does not change what is measured, and the Z/2 table can be
the main one -- with the same barcodes on both sides, so the comparison
is then literally the same computation. If they do not, the Z/47 tables
are measuring something ripser++ does not, and the paper has to say so.

Every run records the foreign load (the cores used by other users), so a
table measured on a busy host is rejected rather than footnoted.

USAGE
    python h2h_coeff_ab.py --reps 5 --out h2h_coeff_ab.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foreign_cpu

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
ASA = "/home/sichaohe/ph_work/ph_dm_1200.npy"
SITE = "/home/sichaohe/canns-lib-dev/site_ship6"
PP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"
# ripser++ reads a Dipha TEXT file, not the .npy: it read the .npy in an
# earlier version of this script and came back with 269 bars in a third of
# a second -- a reference that read nothing, reported as a result.
PP_TXT = "/tmp/asa1200_full.txt"
T = 200.2874965          # max finite distance = the real threshold
MAXDIM = 2

FIELDS = ["rep", "arm", "coeff", "wall_s", "proc_s", "rss_kb", "bars", "md5",
          "h1top", "loadavg", "foreign_cpu", "foreign_max", "verdict"]

CHILD = r'''
import hashlib, json, sys, time
import numpy as np
site, thresh, coeff = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, site)
from canns_lib.ripser import ripser
d = np.load("/home/sichaohe/ph_work/ph_dm_1200.npy")
np.fill_diagonal(d, 0.0)
t0 = time.perf_counter()
r = ripser(d, maxdim=2, thresh=thresh, coeff=coeff, distance_matrix=True,
           progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
rss = 0
for line in open("/proc/self/status"):
    if line.startswith("VmHWM:"):
        rss = int(line.split()[1])
        break
md5 = ":".join(hashlib.md5(np.asarray(sorted(map(tuple, x)), dtype="float64").tobytes()).hexdigest() for x in r)
h1 = np.asarray(r[1], dtype="float64")
life = np.sort(h1[:, 1] - h1[:, 0])[::-1]
life = life[np.isfinite(life)]
print("WALL %.3f" % wall)
print("RSS %d" % rss)
print("BARS " + "|".join(str(len(x)) for x in r))
print("MD5 " + md5)
print("H1TOP " + ",".join("%.2f" % x for x in life[:3]))
'''

# The three arms, in the order they are run. The order ROTATES on every
# repetition, so a machine that drifts during the run moves all three arms
# equally instead of penalising whichever one ran last.
ARMS = [
    ("ours_z47", 47),
    ("ours_z2", 2),
    ("ripserpp", 0),        # 0 = ripser++, which only does Z/2
]


def run_ours(arm, coeff, timeout):
    out_path = "/tmp/_coeffab_%s.out" % arm
    err_path = "/tmp/_coeffab_%s.err" % arm
    env = dict(os.environ)
    env["CANNS_BENCH_GPU"] = "0"
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        with foreign_cpu.Probe() as probe:
            t0 = time.perf_counter()
            proc = subprocess.Popen(
                ["/usr/bin/time", "-f", "__T__%e %M", PY, "-c", CHILD,
                 SITE, repr(T), str(coeff)], stdout=fo, stderr=fe, env=env)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            wall = time.perf_counter() - t0
    rc = proc.returncode
    out = open(out_path).read()
    err = open(err_path).read()
    rec = {"wall_s": "", "proc_s": "", "rss_kb": "", "bars": "", "md5": "",
           "h1top": "", "loadavg": "", "verdict": "RC=%s" % rc}
    for line in out.splitlines():
        if line.startswith("WALL "):
            rec["wall_s"] = line.split()[1]
        elif line.startswith("RSS "):
            rec["rss_kb"] = line.split()[1]
        elif line.startswith("BARS "):
            rec["bars"] = line[5:]
        elif line.startswith("MD5 "):
            rec["md5"] = line[4:]
        elif line.startswith("H1TOP "):
            rec["h1top"] = line[6:]
    m = re.search(r"__T__([\d.]+) (\d+)", err)
    if m:
        rec["proc_s"], rec["rss_kb"] = m.group(1), m.group(2)
    try:
        rec["loadavg"] = "%.2f" % os.getloadavg()[0]
    except OSError:
        rec["loadavg"] = ""
    rec["foreign_cpu"] = probe.record()
    rec["foreign_max"] = probe.record_max()
    if rc == 0 and rec["bars"]:
        rec["verdict"] = "OK"
    return rec


def run_pp(timeout):
    out_path = "/tmp/_coeffab_ripserpp.out"
    err_path = "/tmp/_coeffab_ripserpp.err"
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        with foreign_cpu.Probe() as probe:
            t0 = time.perf_counter()
            proc = subprocess.Popen(
                [PP, "--format", "distance", "--dim", "2", "--threshold",
                 "%.10g" % T, "--ratio", "1.0", PP_TXT],
                stdout=fo, stderr=fe, env=env_gpu())
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            wall = time.perf_counter() - t0
    rc = proc.returncode
    out = open(out_path).read()
    rec = {"wall_s": "", "proc_s": "", "rss_kb": "", "bars": "", "md5": "",
           "h1top": "", "loadavg": "", "verdict": "RC=%s" % rc}
    m = re.search(r"__T__([\d.]+) (\d+)", open(err_path).read())
    if m:
        rec["proc_s"], rec["rss_kb"] = m.group(1), m.group(2)
    # ripser++ prints one interval per line, after a header per dimension
    dims, count = {}, 0
    cur = None
    for line in out.splitlines():
        m2 = re.match(r"persistence intervals in dim (\d+):", line)
        if m2:
            cur = int(m2.group(1))
            dims.setdefault(cur, 0)
            continue
        if cur is not None and line.strip().startswith("["):
            dims[cur] += 1
    if dims:
        rec["bars"] = "|".join(str(dims.get(d, 0)) for d in sorted(dims))
        rec["bars_total"] = sum(dims.values())
        rec["verdict"] = "OK"
        rec["wall_s"] = "%.3f" % wall
    try:
        rec["loadavg"] = "%.2f" % os.getloadavg()[0]
    except OSError:
        rec["loadavg"] = ""
    rec["foreign_cpu"] = probe.record()
    rec["foreign_max"] = probe.record_max()
    return rec


def env_gpu():
    dev = os.environ.get("CANNS_BENCH_GPU", "0")
    return dict(os.environ, CUDA_VISIBLE_DEVICES=dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="h2h_coeff_ab.csv")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    rows = []
    for rep in range(1, args.reps + 1):
        order = ARMS[rep % len(ARMS):] + ARMS[:rep % len(ARMS)]
        for arm, coeff in order:
            rec = (run_pp(args.timeout) if arm == "ripserpp"
                   else run_ours(arm, coeff, args.timeout))
            rec.update({"rep": rep, "arm": arm, "coeff": coeff})
            rows.append(rec)
            print("rep %d  %-9s wall=%-9s bars=%-12s foreign=%s"
                  % (rep, arm, rec["wall_s"], rec["bars"], rec["foreign_cpu"]),
                  flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))

    print("\n| arm | reps | median wall | bars |")
    print("|---|---|---|---|")
    for arm, _c in ARMS:
        walls = [float(r["wall_s"]) for r in rows
                 if r["arm"] == arm and (r.get("wall_s") or "").strip()]
        bars = sorted({r["bars"] for r in rows if r["arm"] == arm})
        if walls:
            print("| %s | %d | %.3f s | %s |"
                  % (arm, len(walls), statistics.median(walls), "/".join(bars)))
    z47 = [float(r["wall_s"]) for r in rows if r["arm"] == "ours_z47"
           and (r.get("wall_s") or "").strip()]
    z2 = [float(r["wall_s"]) for r in rows if r["arm"] == "ours_z2"
          and (r.get("wall_s") or "").strip()]
    if z47 and z2:
        print("\nour Z/47 against our Z/2 (interleaved): %.4fx"
              % (statistics.median(z47) / statistics.median(z2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
