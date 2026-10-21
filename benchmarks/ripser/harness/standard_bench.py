#!/usr/bin/env python3
"""The standard PH benchmark suite, run on all three sides.

WHY THIS EXISTS. Every head-to-head number so far came from ONE matrix
(the 1200x1200 ASA distance matrix), and one matrix is one draw. The
"we beat ripser++" claim needs to hold on the suite ripser++ itself
publishes, because those inputs have a different shape: a different
apparent-pair fraction, different dimension counts, a point cloud
instead of a distance matrix, and much smaller instances.

The suite is the one in the ripser++ source tree
(`testing/run.sh` and `testing/ripser++_testing_expected_barcodes/`):
celegans, dragon1000, HIV, o3_4096, sphere_3_192, Vicsek300. The
expected barcodes are the paper's own reference output, so this script
has THREE independent references per dataset:

  * ripser++ run here, now            (the comparison)
  * ripser++'s published barcodes     (the paper's reference)
  * ours                              (the implementation under test)

and it checks all three against each other, per dimension, with a
tolerance rather than an md5 -- ripser++ prints 6 significant digits, so
an exact hash of its bars can never match ours (see validate_gridcells.py
for why, and check_standard_bench.py for the gate).

THE SIDES
  pp_gpu     ripser++, the CUDA build
  gpu        canns-lib, the `gpu` feature
  cpu        canns-lib, the same source WITHOUT the `gpu` feature
             (the `gpu` feature is opt-in precisely so the released wheel
             builds without nvcc, so this is the build a CPU-only user
             gets)

THE ORDER MATTERS. The three sides are interleaved per dataset and per
repetition, and the minimum over repetitions is reported for each: the
machine is shared, and a side that ran while another tenant was busy
would otherwise be charged for it. The barcode comparison is unaffected
by the order; only the timings are.

USAGE
    python standard_bench.py --gpu-site <site> --cpu-site <site> \
        [--reps 3] [--out standard_bench.csv] [--timeout 3600]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foreign_cpu
from validate_gridcells import match_bars  # the tolerance matcher, shared

PP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"
# The reference CPU implementation, built from the same tree as ripser++.
# It is the fairest CPU baseline available here: the same codebase family,
# the same authors, the same reduction algorithm, on the CPU.
RIPserCPU = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser"
EXAMPLES = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/examples"
EXPECTED = ("/home/sichaohe/ph_work/ripserpp/ripserplusplus/testing/"
            "ripser++_testing_expected_barcodes")
PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"

# key, file, kind, maxdim, ripser++ args, the expected barcode file
DATASETS = [
    ("dragon1000", "dragon1000.distance_matrix", "distance", 2,
     ["--dim", "2"], "dragon1000.gpu.barcodes"),
    ("HIV", "HIV.distance_matrix", "distance", 2,
     ["--dim", "2"], "HIV.gpu.barcodes"),
    ("celegans", "celegans.distance_matrix", "distance", 3,
     ["--dim", "3"], "celegans.gpu.barcodes"),
    ("sphere_3_192", "sphere_3_192.distance_matrix.lower_triangular", "lower_tri", 3,
     ["--dim", "3"], "sphere_3_192.gpu.barcodes"),
    ("Vicsek300", "Vicsek300_300_of_300.distance_matrix", "distance", 3,
     ["--dim", "3"], "Vicsek300_300_of_300.gpu.barcodes"),
    ("o3_4096", "o3_4096.point_cloud", "points", 3,
     ["--threshold", "1.4", "--format", "point-cloud", "--dim", "3", "--sparse"],
     "o3_4096.gpu.barcodes"),
]


def run_timed(argv, stdout_path, timeout, env=None):
    """Run a command with a wall-clock timeout that ACTUALLY fires.

    `/usr/bin/time` is the parent of the program under test, and killing it
    on timeout leaves the program running as an orphan that still holds the
    stderr pipe open -- so `subprocess.run(timeout=...)` waits forever on the
    pipe. The fix is to put the child in its own process group and kill the
    group.
    """
    fo = open(stdout_path, "w")
    proc = subprocess.Popen(argv, stdout=fo, stderr=subprocess.PIPE,
                            text=True, start_new_session=True, env=env)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.communicate()
        fo.close()
        raise
    fo.close()
    return out, err


def run_probed(argv, stdout_path, timeout, env=None):
    """(stdout, stderr, foreign_cpu, foreign_max) for a run.

    How much of the box was used by somebody ELSE while the child ran. The
    load average cannot answer that -- it counts our own threads, which is
    what put it at 3.1 on this 16-core box -- so it is measured directly
    from /proc. `foreign_cpu` is the MEAN number of foreign cores over the
    run (the number the quiet-window gate uses); `foreign_max` is the
    largest rate in any sampling window (a diagnostic). A run that hits the
    timeout still carries its reading: the values are attached to the
    exception on the way out.
    """
    probe = foreign_cpu.Probe()
    try:
        with probe:
            out, err = run_timed(argv, stdout_path, timeout, env=env)
    except subprocess.TimeoutExpired as e:
        try:
            e.foreign = probe.record()
            e.foreign_max = probe.record_max()
        except Exception:
            pass
        raise
    return out, err, probe.record(), probe.record_max()


def gpu_env():
    """The environment for a run that uses the device.

    The box has two A100s and the second one is busy with somebody else's
    work, so the device is pinned rather than left to the default: a
    benchmark that silently landed on a device another job was saturating
    would be a measurement of that job. The device id is recorded in the
    table as well.
    """
    dev = os.environ.get("CANNS_BENCH_GPU", "0")
    return dict(os.environ, CUDA_VISIBLE_DEVICES=dev), dev


def provenance(site):
    """The identity of the build that produced a row.

    The source tree on the benchmark host is a copy, not a git checkout, so
    the commit comes from git when there is one and otherwise from the md5
    of the files the build depends on.
    """
    src = "/home/sichaohe/canns-lib-dev"
    commit = ""
    try:
        commit = subprocess.run(["git", "-C", src, "rev-parse", "--short",
                                 "HEAD"], capture_output=True, text=True,
                                timeout=20).stdout.strip()
    except Exception:
        commit = ""
    if not commit:
        h = hashlib.md5()
        for rel in ("src/ripser/core/reduction.rs", "src/ripser/matrix/sparse.rs",
                    "src/ripser/core/algorithm.rs", "Cargo.toml"):
            try:
                h.update(open(os.path.join(src, rel), "rb").read())
            except OSError:
                pass
        commit = "src:%s" % h.hexdigest()[:12]
    md5 = ""
    if os.path.isdir(site):
        for root, _dirs, names in os.walk(site):
            for name in sorted(names):
                if name.endswith(".so"):
                    try:
                        md5 = hashlib.md5(
                            open(os.path.join(root, name), "rb").read()
                        ).hexdigest()[:16]
                    except OSError:
                        md5 = ""
                    break
            if md5:
                break
    return commit, md5


def binary_md5(path):
    try:
        return hashlib.md5(open(path, "rb").read()).hexdigest()[:16]
    except OSError:
        return ""


def load_dataset(path, kind):
    """Return the array our API takes: a dense distance matrix or points."""
    if kind == "points":
        return np.loadtxt(path, dtype=np.float64)
    if kind == "distance":
        return np.loadtxt(path, dtype=np.float64)
    # lower_triangular: line i holds i values, comma separated, one point
    # per line. ripser++'s reader takes the first i values of line i.
    rows = []
    with open(path) as fh:
        for line in fh:
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


def parse_ripserpp_barcodes(path):
    """{dim: [(birth, death)]} from ripser++'s text output."""
    out, dim = {}, None
    for line in open(path):
        m = re.match(r"persistence intervals in dim (\d+):", line)
        if m:
            dim = int(m.group(1))
            out.setdefault(dim, [])
            continue
        t = line.strip().strip("[]()").split(",")
        if dim is not None and len(t) == 2:
            try:
                out[dim].append((float(t[0]),
                                 float(t[1]) if t[1].strip() else float("inf")))
            except ValueError:
                pass
    return out


def run_pp(key, kind, maxdim, args, timeout):
    """ripser++ on one dataset: returns (wall_s, rss_kb, {dim: bars})."""
    src = os.path.join(EXAMPLES, args["file"])
    env, _dev = gpu_env()
    t0 = time.perf_counter()
    out, err, foreign, foreign_max = run_probed(
        ["/usr/bin/time", "-f", "__T__%e %M", PP] + args["pp_args"] + [src],
        "/tmp/_stdbench_pp.out", timeout, env=env)
    wall = time.perf_counter() - t0
    m = re.search(r"__T__([\d.]+) (\d+)", err)
    rss = int(m.group(2)) if m else 0
    proc = float(m.group(1)) if m else 0.0
    bars = parse_ripserpp_barcodes("/tmp/_stdbench_pp.out")
    return wall, proc, rss, bars, foreign, foreign_max


def run_ripser_cpu(key, kind, maxdim, args, timeout):
    """The reference CPU implementation (the `ripser` binary)."""
    src = os.path.join(EXAMPLES, args["file"])
    # The `ripser` binary takes no `--format`: it reads a full distance
    # matrix, and a point cloud only through its own converter. The one
    # point-cloud dataset is therefore not runnable with it.
    t0 = time.perf_counter()
    out, err, foreign, foreign_max = run_probed(
        ["/usr/bin/time", "-f", "__T__%e %M", RIPserCPU] + args["ripser_cpu_args"]
        + [src], "/tmp/_stdbench_ripser.out", timeout)
    wall = time.perf_counter() - t0
    m = re.search(r"__T__([\d.]+) (\d+)", err)
    rss = int(m.group(2)) if m else 0
    proc = float(m.group(1)) if m else 0.0
    bars = parse_ripserpp_barcodes("/tmp/_stdbench_ripser.out")
    return wall, proc, rss, bars, foreign, foreign_max


def run_ours(site, key, kind, maxdim, args, timeout, coeff):
    """canns-lib on one dataset: returns (wall_s, rss_kb, {dim: bars})."""
    src = os.path.join(EXAMPLES, args["file"])
    child = r'''
import json, sys, time
import numpy as np
site, path, kind, maxdim, thresh, coeff = (sys.argv[1], sys.argv[2], sys.argv[3],
                                            int(sys.argv[4]), float(sys.argv[5]),
                                            int(sys.argv[6]))
sys.path.insert(0, site)
from canns_lib.ripser import ripser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_core
if kind == "points":
    X = np.loadtxt(path, dtype=np.float64)
    dmat = False
else:
    if kind == "lower_tri":
        rows = []
        for line in open(path):
            line = line.strip().rstrip(",")
            rows.append([float(x) for x in line.split(",") if x.strip()])
        n = len(rows)
        X = np.zeros((n, n), dtype=np.float64)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                if j < i:
                    X[i, j] = v
                    X[j, i] = v
    else:
        X = np.loadtxt(path, dtype=np.float64)
    dmat = True
t0 = time.perf_counter()
r = ripser(X, maxdim=maxdim, thresh=thresh, coeff=coeff,
           distance_matrix=dmat, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
bars = {d: [(float(a), float(b)) for a, b in x] for d, x in enumerate(r)}
print("WALL %.3f" % wall)
print("BARS " + json.dumps(bars))
'''
    env, _dev = gpu_env()
    t0 = time.perf_counter()
    out, err, foreign, foreign_max = run_probed(
        ["/usr/bin/time", "-f", "__T__%e %M", PY, "-c", child,
         site, src, kind, str(maxdim),
         "1.4" if key == "o3_4096" else "inf", str(coeff)],
        "/tmp/_stdbench_ours.out", timeout, env=env)
    wall = time.perf_counter() - t0
    m = re.search(r"__T__([\d.]+) (\d+)", err)
    rss = int(m.group(2)) if m else 0
    proc = float(m.group(1)) if m else 0.0
    bars, reported = {}, None
    for line in open("/tmp/_stdbench_ours.out"):
        if line.startswith("WALL "):
            reported = float(line.split()[1])
        elif line.startswith("BARS "):
            bars = {int(k): [tuple(b) for b in v]
                    for k, v in json.loads(line[5:]).items()}
    return (reported if reported is not None else wall, proc, rss, bars,
            foreign, foreign_max)


def compare(a, b):
    """(n_matched, max_rel_dev, unmatched) of bar list a against b."""
    n, dev, ua, ub = match_bars(a, b)
    return n, dev, max(ua, ub)


def row_extra(side, prov_gpu, prov_cpu, pp_md5, cpu_md5, threads, device):
    """The identity of the build that produced a row.

    Our rows carry the source fingerprint, the md5 of the extension that
    was loaded, the thread count and the device. The reference rows carry
    the md5 of the binary that produced them -- the same convention as the
    scale scan and the head-to-head table.
    """
    out = {"loadavg": open("/proc/loadavg").read().split()[0],
           "gpu": device, "threads": "", "commit": "", "build_md5": ""}
    if side == "gpu":
        out["commit"], out["build_md5"] = prov_gpu
        out["threads"] = threads
    elif side == "cpu":
        out["commit"], out["build_md5"] = prov_cpu
        out["threads"] = threads
    elif side == "pp_gpu":
        out["build_md5"] = pp_md5
    elif side == "ripser_cpu":
        out["build_md5"] = cpu_md5
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-site", default="/home/sichaohe/canns-lib-dev/site_ship6")
    ap.add_argument("--cpu-site", default="/home/sichaohe/canns-lib-dev/site_cpu_final")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--coeff", type=int, default=2,
                    help="the coefficient field; 2 matches ripser++")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="standard_bench.csv")
    ap.add_argument("--only", default="",
                    help="comma-separated dataset keys; empty = all six")
    ap.add_argument("--sides", default="",
                    help="comma-separated side names to run; empty = all four. "
                         "The sides are pp_gpu, gpu, cpu, ripser_cpu. Running a "
                         "subset and merging the tables is how the CPU "
                         "reference was added without re-running the GPU arms.")
    args = ap.parse_args()
    only = {k.strip() for k in args.only.split(",") if k.strip()}
    # An EMPTY filter means "every side"; a set of names means "only these".
    sides = {k.strip() for k in args.sides.split(",") if k.strip()} or {
        "pp_gpu", "gpu", "cpu", "ripser_cpu"}

    prov_gpu = provenance(args.gpu_site)
    prov_cpu = provenance(args.cpu_site)
    pp_md5 = binary_md5(PP)
    cpu_md5 = binary_md5(RIPserCPU)
    threads = len(os.sched_getaffinity(0))
    _env, device = gpu_env()
    print("device=%s  commit(gpu)=%s  md5(gpu)=%s  md5(cpu)=%s  threads=%d"
          % (device, prov_gpu[0], prov_gpu[1], prov_cpu[1], threads), flush=True)
    # THE MEASUREMENT LOCK, taken for the whole run. These runners spawn their
    # children directly rather than through bench_core, so the lock has to be
    # taken here: the quiet-window gate counts the cores used by processes
    # that are NOT ours, and it cannot see two of our own measurements
    # running beside each other -- which moves the numbers exactly as much as
    # a neighbour's job does. The lock is released when the process exits.
    _LOCK = bench_core.measurement_lock(label="standard_bench")
    _LOCK.__enter__()

    rows = []
    for key, _fname, kind, maxdim, pp_args, exp_file in DATASETS:
        if only and key not in only:
            continue
        meta = {"key": key, "file": _fname, "kind": kind, "maxdim": maxdim,
                "pp_args": pp_args,
                # The reference CPU binary takes the same switches as
                # ripser++ except `--sparse`, which is ripser++'s own sparse
                # distance-matrix option.
                "ripser_cpu_args": [a for a in pp_args if a != "--sparse"]}
        expected = parse_ripserpp_barcodes(os.path.join(EXPECTED, exp_file))

        # --- the sides, interleaved, every repetition recorded -----------
        # Every repetition is a row of the table. The minimum over
        # repetitions is the best case, and a paper cannot report a best
        # case as the representative value; the median (and the spread
        # around it) is computed by the checker, from these rows.
        per_rep = {}
        pp_bars = ours_bars = cpu_bars = ripser_bars = None
        for rep in range(1, args.reps + 1):
            # --- ripser++ on the GPU -------------------------------------
            if "pp_gpu" in sides:
                try:
                    w, proc, rss, pb, foreign, fmax = run_pp(
                        key, kind, maxdim, meta, args.timeout)
                    per_rep.setdefault("pp_gpu", []).append(
                        (rep, w, proc, rss, foreign, fmax))
                    pp_bars = pb
                    print("%-14s rep%d pp_gpu      %8.3f s" % (key, rep, w), flush=True)
                except subprocess.TimeoutExpired as e:
                    per_rep.setdefault("pp_gpu", []).append(
                        (rep, None, None, None, getattr(e, "foreign", ""),
                         getattr(e, "foreign_max", "")))
                    print("%-14s pp TIMEOUT" % key, flush=True)

            # --- canns-lib on the GPU ------------------------------------
            if "gpu" in sides:
                try:
                    w, proc, rss, ob, foreign, fmax = run_ours(
                        args.gpu_site, key, kind, maxdim, meta, args.timeout,
                        args.coeff)
                    per_rep.setdefault("gpu", []).append(
                        (rep, w, proc, rss, foreign, fmax))
                    ours_bars = ob
                    print("%-14s rep%d canns_gpu   %8.3f s" % (key, rep, w), flush=True)
                except subprocess.TimeoutExpired as e:
                    per_rep.setdefault("gpu", []).append(
                        (rep, None, None, None, getattr(e, "foreign", ""),
                         getattr(e, "foreign_max", "")))
                    print("%-14s canns_gpu TIMEOUT" % key, flush=True)

            # --- the reference CPU implementation -------------------------
            if "ripser_cpu" in sides:
                try:
                    w, proc, rss, rb, foreign, fmax = run_ripser_cpu(
                        key, kind, maxdim, meta, args.timeout)
                    per_rep.setdefault("ripser_cpu", []).append(
                        (rep, w, proc, rss, foreign, fmax))
                    ripser_bars = rb
                    print("%-14s rep%d ripser_cpu  %8.3f s" % (key, rep, w), flush=True)
                except subprocess.TimeoutExpired as e:
                    per_rep.setdefault("ripser_cpu", []).append(
                        (rep, None, None, None, getattr(e, "foreign", ""),
                         getattr(e, "foreign_max", "")))
                    print("%-14s ripser_cpu TIMEOUT" % key, flush=True)

            # --- canns-lib on the CPU ------------------------------------
            if "cpu" in sides:
                try:
                    w, proc, rss, cb, foreign, fmax = run_ours(
                        args.cpu_site, key, kind, maxdim, meta, args.timeout,
                        args.coeff)
                    per_rep.setdefault("cpu", []).append(
                        (rep, w, proc, rss, foreign, fmax))
                    cpu_bars = cb
                    print("%-14s rep%d canns_cpu   %8.3f s" % (key, rep, w), flush=True)
                except subprocess.TimeoutExpired as e:
                    per_rep.setdefault("cpu", []).append(
                        (rep, None, None, None, getattr(e, "foreign", ""),
                         getattr(e, "foreign_max", "")))
                    print("%-14s canns_cpu TIMEOUT" % key, flush=True)

        # --- the barcode comparison, once per side -------------------------
        for side, bars in (("pp_gpu", pp_bars), ("gpu", ours_bars),
                           ("cpu", cpu_bars), ("ripser_cpu", ripser_bars)):
            if sides and side not in sides:
                continue
            reps = per_rep.get(side, [])
            if bars is None or not reps:
                rows.append({"dataset": key, "side": side, "rep": "",
                             "maxdim": maxdim,
                             "wall_s": "", "proc_s": "", "rss_kb": "",
                             "dims": "", "bars_total": "",
                             "match_pp": "", "match_expected": "",
                             "max_rel_dev": "",
                             # A side that did not finish inside the timeout is
                             # a RESULT (it says the implementation cannot do
                             # this instance), not a missing row.
                             "verdict": "TIMEOUT" if reps else "MISSING",
                             **row_extra(side, prov_gpu, prov_cpu, pp_md5,
                                         cpu_md5, threads, device)})
                continue
            dims = sorted(bars)
            total = sum(len(v) for v in bars.values())
            m_pp = m_exp = ""
            devs = []
            if pp_bars is not None:
                unmatched = 0
                for d in dims:
                    # NOT `dev`: that name is the DEVICE, and the comparison
                    # returns a relative deviation -- a shadowing bug once
                    # wrote the deviation into the table's `gpu` column.
                    _n, ddev, un = compare(bars[d], pp_bars.get(d, []))
                    unmatched += un
                    devs.append(ddev)
                m_pp = "yes" if unmatched == 0 else "no"
            if expected:
                # The published expected barcodes only cover the dimensions
                # their generator wrote (dims 0 and 1 -- measured: on every
                # dataset the dim-0 and dim-1 bars match ripser++'s own run
                # here to 0.000e+00, and the file simply has no dim 2/3
                # section). Comparing the dimensions the file does not
                # contain would report a mismatch that is an artefact of the
                # file, not of the implementation.
                unmatched = 0
                devs = []
                for d in sorted(expected):
                    _n, ddev, un = compare(bars.get(d, []), expected[d])
                    unmatched += un
                    devs.append(ddev)
                m_exp = "yes" if unmatched == 0 else "no"
            ok = (m_exp == "yes")
            for (rep, w, proc, rss, foreign, fmax) in reps:
                rows.append({
                    "dataset": key, "side": side, "rep": rep,
                    "maxdim": maxdim,
                    "wall_s": "%.3f" % w if w is not None else "",
                    "proc_s": "%.3f" % proc if proc is not None else "",
                    "rss_kb": rss if rss is not None else "",
                    "foreign_cpu": foreign if foreign is not None else "",
                    "foreign_max": fmax if fmax is not None else "",
                    "dims": "|".join(str(d) for d in dims),
                    "bars_total": total,
                    "match_pp": m_pp, "match_expected": m_exp,
                    "max_rel_dev": "%.3e" % max(devs or [0.0]),
                    "verdict": ("SAME_BARCODE" if ok else "BARCODE_MISMATCH")
                               if w is not None else "TIMEOUT",
                    **row_extra(side, prov_gpu, prov_cpu, pp_md5, cpu_md5,
                                threads, device),
                })
            print("   %-14s %-8s bars=%d match_expected=%s %s"
                  % (key, side, total, m_exp,
                     "SAME_BARCODE" if ok else "BARCODE_MISMATCH"), flush=True)

    fields = ["dataset", "side", "rep", "maxdim", "wall_s", "proc_s", "rss_kb",
              "dims", "bars_total", "match_pp", "match_expected", "max_rel_dev",
              "verdict", "loadavg", "foreign_cpu", "foreign_max", "gpu",
              "threads", "commit", "build_md5"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    # A mismatch is counted per (dataset, side): the repetitions of one side
    # are the same measurement, so a side that mismatched says so once.
    bad = sorted({(r["dataset"], r["side"]) for r in rows
                  if r["verdict"] == "BARCODE_MISMATCH"})
    missing = sorted({(r["dataset"], r["side"]) for r in rows
                      if r["verdict"] in ("MISSING", "TIMEOUT")})
    print("barcode mismatches vs the published reference: %d" % len(bad))
    for ds, side in bad:
        print("   %s / %s" % (ds, side))
    if missing:
        print("sides that did not finish (a result, not a defect): %d" % len(missing))
        for ds, side in missing:
            print("   %s / %s" % (ds, side))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
