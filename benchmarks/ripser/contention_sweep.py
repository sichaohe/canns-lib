#!/usr/bin/env python3
"""How much does a busy host cost each side?

THE QUESTION. The two implementations do their work in different places.
ripser++'s GPU path puts the dimension-2 computation on the device; our
engine reduces on the HOST, with one rayon thread per core, over a column
store that is memory-bound. So a shared cluster is not neutral between
them: a neighbour's job costs us more than it costs ripser++.

That is not a reason to hide the comparison -- it is a deployment property,
and a reader of the paper will ask about it. This script measures the
degradation curve instead of arguing about it: with 0, 4 and 8 competing
CPU processes running, how much slower does each side get?

THE SWEEP GOES UP AND BACK DOWN (0, 4, 8, 8, 4, 0) so that a slow drift
in the machine's own load cancels out of the comparison instead of being
read as an effect of the competing processes.

WHAT THE COMPETING PROCESSES ARE. Busy loops, one per process, each
holding a core. They are not a model of anybody's real job; they are a
controlled amount of CPU taken away from us, which is the quantity the
question is about.

The device is pinned, and the ambient load is recorded on every row, so a
reader can see what else the machine was doing.

USAGE
    python contention_sweep.py --out contention_sweep.csv [--levels 0,4,8]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import time

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
PP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"
ASA_NPY = "/home/sichaohe/ph_work/ph_dm_1200.npy"

REF_MD5 = ("ee1f5283091dd645a3dd53e6f1a7eac6:"
           "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
           "2d018d8fe9c1895b1b74f837af59ed20")
REF_BARS = "1195|481|58"

FIELDS = ["pass", "level", "side", "rep", "wall_s", "proc_s", "rss_kb", "bars",
          "md5", "loadavg", "gpu", "verdict"]

CHILD = r'''
import hashlib, sys, time
import numpy as np
site, path, maxdim, coeff = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                             int(sys.argv[4]))
sys.path.insert(0, site)
from canns_lib.ripser import ripser
d = np.load(path)
np.fill_diagonal(d, 0.0)
t0 = time.perf_counter()
r = ripser(d, maxdim=maxdim, thresh=float("inf"), coeff=coeff,
           distance_matrix=True, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
md5 = ":".join(hashlib.md5(np.asarray(sorted(map(tuple, x)), dtype="float64").tobytes()).hexdigest() for x in r)
rss = 0
for line in open("/proc/self/status"):
    if line.startswith("VmHWM:"):
        rss = int(line.split()[1])
        break
print("WALL %.3f" % wall)
print("RSS %d" % rss)
print("BARS " + "|".join(str(len(x)) for x in r))
print("MD5 " + md5)
'''


def busy():
    """A process that holds a core."""
    x = 0
    while True:
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF


def start_competitors(n):
    procs = []
    for _ in range(n):
        p = multiprocessing.Process(target=busy, daemon=True)
        p.start()
        procs.append(p)
    return procs


def stop_competitors(procs):
    for p in procs:
        try:
            os.kill(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


def run_ours(site, path, maxdim, coeff, timeout, env):
    t0 = time.perf_counter()
    p = subprocess.run(["/usr/bin/time", "-f", "__T__%e %M", PY, "-c", CHILD,
                        site, path, str(maxdim), str(coeff)],
                       capture_output=True, text=True, env=env, timeout=timeout)
    out = {}
    for line in p.stdout.splitlines():
        if " " in line:
            k, _, v = line.partition(" ")
            out[k] = v
    # the child prints upper-case keys; the table uses lower-case ones
    out["bars"] = out.get("BARS", "")
    out["md5"] = out.get("MD5", "")
    m = re.search(r"__T__([\d.]+) (\d+)", p.stderr)
    out["proc_s"] = m.group(1) if m else ""
    out["rss_kb"] = m.group(2) if m else ""
    out["rc"] = p.returncode
    out["host_s"] = "%.3f" % (time.perf_counter() - t0)
    return out


def run_reference(path, maxdim, timeout, env, thresh="inf"):
    """The reference, on the capped matrix.

    The default threshold is the maximum finite distance -- the FAIR arm,
    the one the paper compares against: it excludes exactly the pairs we
    keep as `+inf`, so the two sides compute the same filtration. The cap
    as the threshold is the control and is NOT what this experiment
    measures.
    """
    t0 = time.perf_counter()
    p = subprocess.run(["/usr/bin/time", "-f", "__T__%e %M", PP,
                        "--format", "distance", "--dim", str(maxdim),
                        "--threshold", thresh, "--ratio", "1.0", path],
                       capture_output=True, text=True, env=env, timeout=timeout)
    m = re.search(r"__T__([\d.]+) (\d+)", p.stderr)
    out = {"proc_s": m.group(1) if m else "", "rss_kb": m.group(2) if m else "",
           "rc": p.returncode, "host_s": "%.3f" % (time.perf_counter() - t0)}
    # per-dimension bar counts, the same shape as the reference barcode
    per, dim = {}, None
    for ln in p.stdout.splitlines():
        mm = re.match(r"persistence intervals in dim (\d+):", ln)
        if mm:
            dim = int(mm.group(1))
            per.setdefault(dim, 0)
        elif dim is not None and ln.startswith(" ["):
            per[dim] += 1
    out["bars"] = "|".join(str(per[d]) for d in sorted(per))
    out["md5"] = "n/a"   # the reference prints 6 digits; it has no md5
    out["WALL"] = out["proc_s"]
    return out


def write_capped_text(path, npy_path):
    """The capped matrix as text, for the reference.

    ripser++'s reader takes text and cannot express `+inf`, so the
    reference gets the capped matrix. The bar COUNTS are the same as the
    uncapped one's (the cap does not change which simplices are in the
    filtration), so the barcode gate still compares like with like.
    """
    import numpy as np
    d = np.load(npy_path)
    fin = d[np.isfinite(d)]
    cap = float(fin.max()) * 2 + 1.0
    np.savetxt(path, np.where(np.isfinite(d), d, cap), fmt="%.10g")
    return path, float(fin.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/home/sichaohe/canns-lib-dev/site_ship6")
    ap.add_argument("--out", default="contention_sweep.csv")
    ap.add_argument("--levels", default="0,4,8")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--coeff", type=int, default=47)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds to wait after starting the competitors")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    # up and back down, so that a drift in the machine's own load cancels
    order = levels if len(levels) < 2 else levels + list(reversed(levels[:-1]))
    txt_path, T = write_capped_text("/tmp/_cont_asa_capped.txt", ASA_NPY)
    dev = os.environ.get("CANNS_BENCH_GPU", "0")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev)
    print("device=%s  levels=%s  sweep order=%s  reference threshold=T=%.6g"
          % (dev, levels, order, T), flush=True)

    rows = []
    for step, level in enumerate(order):
        # the sweep goes up and back down, so the same level appears twice
        # with the same repetition numbers; the pass tells the two apart
        pass_no = 1 if step < len(levels) else 2
        procs = start_competitors(level)
        time.sleep(args.settle)
        for rep in range(1, args.reps + 1):
            for side in ("ours", "ripserpp"):
                if side == "ours":
                    r = run_ours(args.site, ASA_NPY, args.maxdim, args.coeff,
                                 args.timeout, env)
                else:
                    r = run_reference(txt_path, args.maxdim, args.timeout,
                                      env, thresh="%.10g" % T)
                ok = (r.get("rc") == 0 and r.get("bars") == REF_BARS
                      and (side == "ripserpp" or r.get("md5") == REF_MD5))
                rows.append({
                    "pass": pass_no, "level": level, "side": side, "rep": rep,
                    "wall_s": r.get("WALL", ""), "proc_s": r.get("proc_s", ""),
                    "rss_kb": r.get("rss_kb", ""), "bars": r.get("bars", ""),
                    "md5": r.get("md5", ""),
                    "loadavg": open("/proc/loadavg").read().split()[0],
                    "gpu": dev,
                    "verdict": "OK" if ok else "BADCODE",
                })
                print("pass=%d level=%-2d %-9s rep%d wall=%-8s bars=%-12s load=%-6s %s"
                      % (pass_no, level, side, rep, r.get("WALL", ""), r.get("bars", ""),
                         rows[-1]["loadavg"], rows[-1]["verdict"]), flush=True)
        stop_competitors(procs)
        time.sleep(1.0)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for x in rows:
                w.writerow(x)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for x in rows:
            w.writerow(x)
    bad = [r for r in rows if r["verdict"] != "OK"]
    print("\nwrote %s (%d rows); runs that did not reproduce the barcode: %d"
          % (args.out, len(rows), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
