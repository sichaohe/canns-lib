#!/usr/bin/env python3
"""Where does the time go on o3_4096, and why are we slower than the reference?

o3_4096 is the one dataset in ripser++'s own suite where we lose badly: on
the GPU build we are 3.6x ripser++'s time, and on the CPU build we are 1.58x
the single-threaded classic Ripser's -- which is the more damning number,
because that reference uses one thread and we use sixteen.

This script produces the evidence for WHY, before anybody writes code:

* our own build prints a phase breakdown per dimension when
  `CANNS_RIPSER_STATS` is set (the instrumented lines are compiled into the
  shipping build; no special build is needed);
* a patched COPY of ripser++ prints the same shape of numbers
  (`REDUCEPROF`), which is what the earlier `reduce_split.csv` table used;
* the pristine ripser++ is run as well, for the wall-clock reference, because
  the patched build carries counters and is therefore not the thing whose
  time we quote.

Every run records the foreign load (see foreign_cpu.py): the numbers are
only worth reading if the box was quiet while they were taken.

The output is one row per (side, dimension). The columns that MUST agree
between the two sides are the operation counts -- columns reduced, reduction
rounds, peak coboundary heap -- because if they agree and the times do not,
the difference is in the implementation and not in the algorithm.

USAGE
    python o3_profile.py --out o3_profile.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foreign_cpu
# The measurement lock lives in bench_core, and this import has to be in the
# PARENT's scope. The one inside the child script string further down is the
# child's: the parent never gets it, so `bench_core.measurement_lock` raised
# NameError and the run died before it measured anything.
import bench_core

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
EXAMPLES = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/examples"
PP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"
PP_PROFILED = "/home/sichaohe/ph_work/ripserpp/rp_count/build/ripser++"
RIPserCPU = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser"
GPU_SITE = "/home/sichaohe/canns-lib-dev/site_ship6"
CPU_SITE = "/home/sichaohe/canns-lib-dev/site_cpu_final"

# o3_4096, exactly as the classical suite runs it: a point cloud, dimension
# 3, threshold 1.4, coefficient 2.
DATASET = "o3_4096"
FILE = "o3_4096.point_cloud"
MAXDIM = 3
THRESH = 1.4
COEFF = 2
PP_ARGS = ["--threshold", "1.4", "--format", "point-cloud", "--dim", "3",
           "--sparse"]

FIELDS = ["side", "dim", "columns", "apparent", "nonapparent_cols",
          "ratio_filter_survives", "rounds", "heap_max", "col_rounds_max",
          "cofacets", "scan_s", "assemble_s", "gpu_scan_s", "fill_s",
          "reduce_s", "push_s", "pop_s", "loop_s", "gpu_pre_s", "densify_s",
          "pre_loop_s", "pivots", "dim_total_s", "wall_s", "rss_kb", "bars",
          "foreign_cpu", "foreign_max", "verdict"]


def gpu_env():
    """The device is pinned: the second GPU on this box is somebody else's."""
    dev = os.environ.get("CANNS_BENCH_GPU", "0")
    return dict(os.environ, CUDA_VISIBLE_DEVICES=dev), dev


CHILD = r'''
import json, sys, time
import numpy as np
site, path, maxdim, thresh, coeff = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                     float(sys.argv[4]), int(sys.argv[5]))
sys.path.insert(0, site)
from canns_lib.ripser import ripser
X = np.loadtxt(path, dtype=np.float64)
t0 = time.perf_counter()
r = ripser(X, maxdim=maxdim, thresh=thresh, coeff=coeff,
           distance_matrix=False, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
rss = 0
for line in open("/proc/self/status"):
    if line.startswith("VmHWM:"):
        rss = int(line.split()[1])
        break
print("WALL %.3f" % wall)
print("RSS %d" % rss)
print("BARS " + json.dumps({str(d): len(x) for d, x in enumerate(r)}))
'''


def kv(line):
    """The key=value pairs of one instrumented line."""
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


def num(d, key, cast=float):
    try:
        return cast(d.get(key, ""))
    except (TypeError, ValueError):
        return ""


def run_ours(side, site, timeout):
    """Our build on o3, with the instrumented lines in its stderr."""
    env, dev = gpu_env()
    env["CANNS_RIPSER_STATS"] = "1"
    src = os.path.join(EXAMPLES, FILE)
    out_path = "/tmp/_o3prof_%s.out" % side
    err_path = "/tmp/_o3prof_%s.err" % side
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        with foreign_cpu.Probe() as probe:
            t0 = time.perf_counter()
            proc = subprocess.Popen(
                [PY, "-c", CHILD, site, src, str(MAXDIM), str(THRESH),
                 str(COEFF)], stdout=fo, stderr=fe, env=env)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            wall = time.perf_counter() - t0
    rc = proc.returncode
    out = open(out_path).read()
    err = open(err_path).read()
    bars = {}
    for line in out.splitlines():
        if line.startswith("BARS "):
            bars = json.loads(line[5:])
    rows = []
    for line in err.splitlines():
        if not line.startswith("STATS dim="):
            continue
        d = kv(line)
        rows.append({
            "side": side,
            "dim": num(d, "dim", int),
            "columns": num(d, "columns", int),
            "apparent": num(d, "apparent_pairs", int),
            "nonapparent_cols": num(d, "nonapparent_cols", int),
            "ratio_filter_survives": num(d, "ratio_filter_survives"),
            "rounds": num(d, "rounds", int),
            "heap_max": num(d, "heap_max", int),
            "col_rounds_max": num(d, "col_rounds_max", int),
            "cofacets": num(d, "cofacets_per_col", float),
            "scan_s": num(d, "scan_s"),
            "assemble_s": "",
            "gpu_scan_s": "",
            "fill_s": num(d, "fill_s"),
            "reduce_s": num(d, "reduce_s"),
            "push_s": num(d, "push_s"),
            "pop_s": num(d, "pop_s"),
            "loop_s": num(d, "loop_s"),
            "gpu_pre_s": num(d, "gpu_pre_s"),
            "densify_s": num(d, "densify_s"),
            "pre_loop_s": num(d, "pre_loop_s"),
            "pivots": num(d, "pivots", int),
            "dim_total_s": num(d, "total_s"),
            "wall_s": "%.3f" % wall,
            "rss_kb": "",
            "bars": "".join("%s:%s|" % (k, v) for k, v in sorted(bars.items())),
            "foreign_cpu": probe.record(),
            "foreign_max": probe.record_max(),
            "verdict": "OK" if rc == 0 and bars else "RC=%s" % rc,
        })
    return rows


def run_pp(side, binary, timeout, profiled):
    """ripser++ on o3; the patched build adds the per-dimension breakdown."""
    env, dev = gpu_env()
    src = os.path.join(EXAMPLES, FILE)
    out_path = "/tmp/_o3prof_%s.out" % side
    err_path = "/tmp/_o3prof_%s.err" % side
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        with foreign_cpu.Probe() as probe:
            t0 = time.perf_counter()
            proc = subprocess.Popen([binary] + PP_ARGS + [src], stdout=fo,
                                    stderr=fe, env=env)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            wall = time.perf_counter() - t0
    rc = proc.returncode
    err = open(err_path).read()
    rows = []
    # ripser++ prints its own per-dimension breakdown; the REDUCEPROF line
    # is only in the patched copy, and even that copy did not print it on
    # this dataset, so the numbers are taken from the pristine output.
    per_dim = defaultdict(dict)
    total_time = ""
    for line in err.splitlines():
        m = re.search(r"columns to reduce for dim: (\d+): (\d+)", line)
        if m:
            per_dim[int(m.group(1))]["columns"] = m.group(2)
            continue
        m = re.search(r"num cols to reduce: dim (\d+), (\d+)", line)
        if m:
            # dimension 1 is printed in this shape, not the other one
            per_dim.setdefault(int(m.group(1)), {})["columns"] = m.group(2)
            continue
        m = re.search(r"num apparent for dim: (\d+) is: (\d+)", line)
        if m:
            per_dim[int(m.group(1))]["apparent"] = m.group(2)
            continue
        m = re.search(r"SUBMATRIX REDUCTION TIME for dim (\d+): ([0-9.]+)s", line)
        if m:
            per_dim[int(m.group(1))]["reduce_s"] = m.group(2)
            continue
        m = re.search(r"ASSEMBLE COLS TIME for dim (\d+): ([0-9.]+)s", line)
        if m:
            per_dim[int(m.group(1))]["assemble_s"] = m.group(2)
            continue
        m = re.search(r"GPU MATRIX SCAN and post processing time for dim (\d+): ([0-9.]+)s", line)
        if m:
            per_dim[int(m.group(1))]["gpu_scan_s"] = m.group(2)
            continue
        m = re.search(r"^total time: ([0-9.]+)s", line)
        if m:
            total_time = m.group(1)
    for dim in sorted(per_dim):
        d = per_dim[dim]
        rows.append({
            "side": side,
            "dim": dim,
            "columns": d.get("columns", ""),
            "apparent": d.get("apparent", ""),
            "nonapparent_cols": "",
            "ratio_filter_survives": "",
            "rounds": "",
            "heap_max": "",
            "col_rounds_max": "",
            "cofacets": "",
            "scan_s": "",
            "assemble_s": d.get("assemble_s", ""),
            "gpu_scan_s": d.get("gpu_scan_s", ""),
            "fill_s": "",
            "reduce_s": d.get("reduce_s", ""),
            "push_s": "",
            "pop_s": "",
            "loop_s": "",
            "gpu_pre_s": "",
            "densify_s": "",
            "pre_loop_s": "",
            "pivots": "",
            "dim_total_s": "",
            "wall_s": "%.3f" % wall,
            "rss_kb": "",
            "bars": "",
            "foreign_cpu": probe.record(),
            "foreign_max": probe.record_max(),
            "verdict": "OK" if rc == 0 else "RC=%s" % rc,
        })
    if not rows:
        # The pristine build prints no breakdown; it still gives the wall time.
        rows.append({
            "side": side, "dim": "", "columns": "", "apparent": "",
            "nonapparent_cols": "", "ratio_filter_survives": "",
            "rounds": "", "heap_max": "", "col_rounds_max": "", "cofacets": "",
            "scan_s": "", "assemble_s": "", "gpu_scan_s": "", "fill_s": "",
            "reduce_s": "", "push_s": "", "pop_s": "", "loop_s": "",
            "gpu_pre_s": "", "densify_s": "", "pre_loop_s": "", "pivots": "",
            "dim_total_s": "", "wall_s": "%.3f" % wall,
            "rss_kb": "", "bars": "", "foreign_cpu": probe.record(),
            "foreign_max": probe.record_max(),
            "verdict": "OK" if rc == 0 else "RC=%s" % rc,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="o3_profile.csv")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--sides", default="ours_gpu,ours_cpu,pp_profiled,pp")
    args = ap.parse_args()
    # THE MEASUREMENT LOCK, taken for the whole run. These runners spawn their
    # children directly rather than through bench_core, so the lock has to be
    # taken here: the quiet-window gate counts the cores used by processes
    # that are NOT ours, and it cannot see two of our own measurements
    # running beside each other -- which moves the numbers exactly as much as
    # a neighbour's job does. The lock is released when the process exits.
    _LOCK = bench_core.measurement_lock(label="o3_profile")
    _LOCK.__enter__()

    rows = []
    for side in [s.strip() for s in args.sides.split(",") if s.strip()]:
        print("--- %s" % side, flush=True)
        if side == "ours_gpu":
            rows += run_ours(side, GPU_SITE, args.timeout)
        elif side == "ours_cpu":
            rows += run_ours(side, CPU_SITE, args.timeout)
        elif side == "pp_profiled":
            rows += run_pp(side, PP_PROFILED, args.timeout, True)
        elif side == "pp":
            rows += run_pp(side, PP, args.timeout, False)
        else:
            print("unknown side %s" % side, file=sys.stderr)
            return 2

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))

    print("\n| side | dim | columns | apparent | scan | assemble | gpu scan | reduce | wall |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
              % (r["side"], r["dim"], r["columns"], r["apparent"],
                 r["scan_s"], r["assemble_s"], r["gpu_scan_s"],
                 r["reduce_s"], r["wall_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
