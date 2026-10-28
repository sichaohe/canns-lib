#!/usr/bin/env python3
"""Why is our CPU build slower than the single-threaded classic Ripser?

On o3_4096 our CPU build takes ~94 s and the classic `ripser` C++ binary
takes ~59 s -- and the classic one uses ONE thread while ours uses all
sixteen. That gap is not a device difference (both run on the host), so it
has to be either the inner loop or the fact that the parallelism does not
pay for itself on this workload.

This script separates the two:

* **our side**, at several thread counts (`RAYON_NUM_THREADS` is what
  Rayon reads), with the per-dimension phase breakdown the build prints
  behind `CANNS_RIPSER_STATS`. If one thread is already as slow as
  sixteen, the parallelism is not the problem; if sixteen threads buy
  nothing over one, the parallelism is the problem.
* **the classic side**, once, with every line of its output timestamped.
  It prints "assembling columns", "sorting N columns to reduce" and
  "reducing column X of M" as it goes, so the per-dimension split between
  assembly and reduction can be recovered from the timeline. Its source is
  never modified.

Raw table: `o3_cpu_scaling.csv`, one row per (side, threads, dimension).
Checker: `check_o3_cpu_scaling.py`.

USAGE
    python o3_cpu_scaling.py --out o3_cpu_scaling.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foreign_cpu
# The measurement lock lives in bench_core, and this import has to be in the
# PARENT's scope. The one inside the child script string further down is the
# child's: the parent never gets it, so `bench_core.measurement_lock` raised
# NameError and the run died before it measured anything.
import bench_core

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
EXAMPLES = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/examples"
RIPserCPU = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser"
CPU_SITE = "/home/sichaohe/canns-lib-dev/site_cpu_final"

FILE = "o3_4096.point_cloud"
MAXDIM = 3
THRESH = 1.4
COEFF = 2

FIELDS = ["side", "threads", "dim", "stage", "value_s", "columns", "rounds",
          "push_s", "pop_s", "scan_s", "fill_s", "reduce_s", "total_s",
          "wall_s", "foreign_cpu", "foreign_max", "verdict"]

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
print("WALL %.3f" % wall)
print("BARS " + json.dumps({str(d): len(x) for d, x in enumerate(r)}))
'''


def kv(line):
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


def run_ours(threads, timeout):
    env = dict(os.environ)
    env["CANNS_RIPSER_STATS"] = "1"
    env["RAYON_NUM_THREADS"] = str(threads)
    src = os.path.join(EXAMPLES, FILE)
    out_path = "/tmp/_o3cpu_%d.out" % threads
    err_path = "/tmp/_o3cpu_%d.err" % threads
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        with foreign_cpu.Probe() as probe:
            t0 = time.perf_counter()
            proc = subprocess.Popen([PY, "-c", CHILD, CPU_SITE, src,
                                     str(MAXDIM), str(THRESH), str(COEFF)],
                                    stdout=fo, stderr=fe, env=env)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            wall = time.perf_counter() - t0
    rc = proc.returncode
    err = open(err_path).read()
    rows = []
    for line in err.splitlines():
        if not line.startswith("STATS dim="):
            continue
        d = kv(line)
        rows.append({
            "side": "ours", "threads": threads, "dim": num(d, "dim", int),
            "stage": "phases",
            "value_s": num(d, "total_s"),
            "columns": num(d, "columns", int),
            "rounds": num(d, "rounds", int),
            "push_s": num(d, "push_s"), "pop_s": num(d, "pop_s"),
            "scan_s": num(d, "scan_s"), "fill_s": num(d, "fill_s"),
            "reduce_s": num(d, "reduce_s"), "total_s": num(d, "total_s"),
            "wall_s": "%.3f" % wall, "foreign_cpu": probe.record(),
            "foreign_max": probe.record_max(),
            "verdict": "OK" if rc == 0 else "RC=%s" % rc,
        })
    return rows, wall, rc


def run_classic(timeout):
    """The classic Ripser, with every line of its output timestamped."""
    src = os.path.join(EXAMPLES, FILE)
    args = [RIPserCPU, "--threshold", str(THRESH), "--format", "point-cloud",
            "--dim", str(MAXDIM), src]
    timeline = []
    t0 = time.perf_counter()
    with foreign_cpu.Probe() as probe:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        for line in proc.stdout:
            timeline.append((time.perf_counter() - t0, line.rstrip("\n")))
        proc.wait(timeout=timeout)
        wall = time.perf_counter() - t0
    rc = proc.returncode

    # The timeline: "assembling columns" starts a dimension, "sorting N
    # columns to reduce" ends its assembly, and the "reducing column X of
    # M" messages carry the reduction's progress.
    rows = []
    assemble_start = None
    sort_at = None
    dim = None
    last_reduce_at = None
    total_cols = None
    for t, line in timeline:
        clean = line.replace("\r", " ").strip()
        if "assembling columns" in clean:
            if dim is not None and sort_at is not None and last_reduce_at:
                rows.append(_classic_row(dim, assemble_start, sort_at,
                                         last_reduce_at, total_cols, wall,
                                         probe))
            assemble_start, sort_at, last_reduce_at, total_cols = t, None, None, None
            dim = (dim + 1) if dim is not None else 1
        m = re.search(r"sorting (\d+) columns to reduce", clean)
        if m:
            sort_at, total_cols = t, m.group(1)
        m = re.search(r"reducing column (\d+) of (\d+)", clean)
        if m:
            last_reduce_at = t
            total_cols = m.group(2)
    if dim is not None and sort_at is not None and last_reduce_at:
        rows.append(_classic_row(dim, assemble_start, sort_at, last_reduce_at,
                                 total_cols, wall, probe))
    # A row for the whole run
    rows.append({
        "side": "classic", "threads": 1, "dim": "", "stage": "total",
        "value_s": "%.3f" % wall, "columns": "", "rounds": "", "push_s": "",
        "pop_s": "", "scan_s": "", "fill_s": "", "reduce_s": "",
        "total_s": "%.3f" % wall, "wall_s": "%.3f" % wall,
        "foreign_cpu": probe.record(), "foreign_max": probe.record_max(),
        "verdict": "OK" if rc == 0 else "RC=%s" % rc,
    })
    return rows, wall, rc, timeline


def _classic_row(dim, assemble_start, sort_at, last_reduce_at, total_cols,
                 wall, probe):
    return {
        "side": "classic", "threads": 1, "dim": dim, "stage": "phases",
        "value_s": "%.3f" % (last_reduce_at - assemble_start),
        "columns": total_cols or "", "rounds": "", "push_s": "", "pop_s": "",
        "scan_s": "%.3f" % (sort_at - assemble_start), "fill_s": "",
        "reduce_s": "%.3f" % (last_reduce_at - sort_at),
        "total_s": "%.3f" % (last_reduce_at - assemble_start),
        "wall_s": "%.3f" % wall, "foreign_cpu": probe.record(),
        "foreign_max": probe.record_max(), "verdict": "OK",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="o3_cpu_scaling.csv")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--threads", default="1,2,4,16",
                    help="the thread counts to measure our build at")
    ap.add_argument("--only-classic", action="store_true",
                    help="measure only the reference implementation; used to "
                         "re-measure it without repeating our own runs")
    args = ap.parse_args()
    # THE MEASUREMENT LOCK, taken for the whole run. These runners spawn their
    # children directly rather than through bench_core, so the lock has to be
    # taken here: the quiet-window gate counts the cores used by processes
    # that are NOT ours, and it cannot see two of our own measurements
    # running beside each other -- which moves the numbers exactly as much as
    # a neighbour's job does. The lock is released when the process exits.
    _LOCK = bench_core.measurement_lock(label="o3_cpu_scaling")
    _LOCK.__enter__()

    rows = []
    print("=== the classic Ripser (one thread) ===", flush=True)
    classic, cwall, crc, timeline = run_classic(args.timeout)
    rows += classic
    print("  wall %.3f s   rc=%s   (%d output lines)"
          % (cwall, crc, len(timeline)), flush=True)

    for t in ([] if args.only_classic
              else [int(x) for x in args.threads.split(",") if x.strip()]):
        print("=== ours, %d thread(s) ===" % t, flush=True)
        ours, wall, rc = run_ours(t, args.timeout)
        rows += ours
        print("  wall %.3f s   rc=%s" % (wall, rc), flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))

    print("\n| side | threads | dim | columns | scan/assemble | reduce | total |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %s | %s | %s | %s | %s |"
              % (r["side"], r["threads"], r["dim"], r["columns"], r["scan_s"],
                 r["reduce_s"], r["total_s"]))
    print("\nwall clock:")
    for r in rows:
        if r["stage"] == "total" or r["dim"] == 1:
            print("  %-8s %2s threads: %s s" % (r["side"], r["threads"],
                                                r["wall_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
