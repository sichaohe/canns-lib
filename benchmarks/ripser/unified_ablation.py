#!/usr/bin/env python3
"""The unified ablation table: one switch per row, one pass, one protocol.

WHAT THIS IS. The paper's central table is an ablation: for each design
decision, what did it buy? The measurements for the individual decisions
exist, but they were taken in different rounds, on different days, with
different harnesses -- which is not a table a reader can read as a table.

This script measures them ALL AGAIN, in one interleaved pass, with one
protocol, so that the rows are comparable with each other and not only
within themselves. For each change it runs the build BEFORE the change and
the build AFTER it, back to back, on the same arm, in the same repetition.

THE FOUR CHANGES, each with its own before/after pair of builds:

  auto_sparse        the auto-sparse decision no longer sends a dense
                     matrix down the sparse path (the ASA pipeline
                     replaces the +inf pairs with a cap, so the old
                     `threshold < max_finite` test was true for an
                     ordinary dense matrix, and the sparse path is 4.3x
                     slower on identical work).
  coboundary_heap    the coboundary heap pops with a classic sift-down
                     instead of `BinaryHeap::pop`.
  nonempty_bounds    the column store keeps one boundary pair per
                     NON-EMPTY column instead of one per enumerated
                     column.
  zero_copy_handoff  the GPU pre-pass hands the column indices to the
                     host without copying them twice.

THE ARMS. The two input forms x the two thresholds, because the changes
interact with both: the `+inf` matrix at the real threshold (the dense
path), the `+inf` matrix at the cap, the capped matrix at the real
threshold (ripser++'s input -- the arm the sparse false positive used to
ruin), and the capped matrix at the cap.

GATES. Every run must print the reference barcode md5 and the bar counts.
A run that disagrees is recorded as `BADCODEC` and the checker fails the
whole table: an ablation row that changed the answer is not an ablation
row, it is a bug.

⚠ EVERY BUILD MUST BE COMPILED WITH THE `gpu` FEATURE. The GPU apparent
  pre-pass lives behind `#[cfg(feature = "gpu")]` and the feature is not
  in the default set, so a build without it silently falls back to the
  CPU pre-pass and is ~35% slower -- which reads exactly like a
  regression in the change under test. The script refuses to run against
  a build whose `gpu_err` is not empty.

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

# The changes, in the order the paper tells them, with the build pair of
# each. The `before` build is the one WITHOUT the change.
CHANGES = [
    ("auto_sparse", "orig", "/home/sichaohe/canns-lib-dev/site_g_orig",
     "modp", "/home/sichaohe/canns-lib-dev/site_g_modp"),
    ("coboundary_heap", "heap_off", "/home/sichaohe/canns-lib-dev/site_hp_off",
     "heap_on", "/home/sichaohe/canns-lib-dev/site_hp_on"),
    ("nonempty_bounds", "ship", "/home/sichaohe/canns-lib-dev/site_ship",
     "bounds", "/home/sichaohe/canns-lib-dev/site_bounds"),
    ("zero_copy_handoff", "ship3", "/home/sichaohe/canns-lib-dev/site_ship3",
     "ship4", "/home/sichaohe/canns-lib-dev/site_ship4"),
]

FIELDS = ["rep", "change", "arm", "role", "build", "mode", "thresh",
          "wall_s", "rss_kb", "bars", "md5", "h1top", "gpu_err", "rc",
          "verdict", "loadavg", "gpu", "commit"]

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


def source_commit():
    """The fingerprint of the source the builds came from.

    The tree on the benchmark host is a copy, not a git checkout, so the
    commit comes from git when there is one and otherwise from the md5 of
    the files the build depends on.
    """
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
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of arm names (default: all four)")
    ap.add_argument("--changes", nargs="*", default=None,
                    help="subset of change names (default: all four)")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    arms = [a for a in ARMS if args.arms is None or a[0] in args.arms]
    changes = [c for c in CHANGES if args.changes is None or c[0] in args.changes]
    if not arms or not changes:
        print("FAIL: nothing selected", file=sys.stderr)
        return 2

    # The device is pinned: the box has two A100s and the second one
    # belongs to somebody else's job, and a run that landed on a device
    # another job was saturating would be a measurement of that job.
    dev = os.environ.get("CANNS_BENCH_GPU", "0")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev)
    commit = source_commit()
    print("device=%s  commit=%s  reps=%d  changes=%s"
          % (dev, commit, args.reps, [c[0] for c in changes]), flush=True)

    rows = []
    for rep in range(1, args.reps + 1):
        for change, bname, bpath, aname, apath in changes:
            for arm, mode, thresh in arms:
                # before and after, back to back, so the machine state is
                # as close to the same for the two as it can be
                for role, bld, path in (("before", bname, bpath),
                                        ("after", aname, apath)):
                    t0 = time.time()
                    r = run(path, thresh, mode, args.timeout, env=env)
                    rec = {
                        "rep": rep, "change": change, "arm": arm, "role": role,
                        "build": bld, "mode": mode, "thresh": thresh,
                        "wall_s": r.get("WALL", ""),
                        "rss_kb": r.get("RSS", ""),
                        "bars": r.get("BARS", ""),
                        "md5": r.get("MD5", ""),
                        "h1top": r.get("H1TOP", ""),
                        "gpu_err": r.get("gpu_err", ""),
                        "rc": r.get("rc", ""),
                        "loadavg": open("/proc/loadavg").read().split()[0],
                        "gpu": dev, "commit": commit,
                    }
                    if rec["gpu_err"]:
                        rec["verdict"] = "NOGPU"
                    elif rec["md5"] != REF_MD5 or rec["bars"] != REF_BARS:
                        rec["verdict"] = "BADCODEC"
                    else:
                        rec["verdict"] = "OK"
                    rows.append(rec)
                    print("rep=%d %-18s %-8s %-6s %-8s wall=%-8s rss=%-9s %s %s"
                          % (rep, change, arm, role, bld, rec["wall_s"],
                             rec["rss_kb"], rec["bars"], rec["verdict"]),
                          flush=True)
        write_table(args.out, rows)

    write_table(args.out, rows)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    bad = sorted({(r["change"], r["arm"], r["build"]) for r in rows
                  if r["verdict"] != "OK"})
    print("runs that did not reproduce the reference barcode: %d" % len(bad))
    for c in bad:
        print("   %s / %s / %s" % c)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
