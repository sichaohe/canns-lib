#!/usr/bin/env python
"""Same-batch gap accounting: canns-lib vs ripser++, stage by stage.

WHY THIS EXISTS. The phase-1 gap table was built from two different batches
on a loaded machine, which is exactly the comparison that produced a 14%
drift and a wrong conclusion earlier in this project. Every number in the
output below comes from ONE interleaved batch.

THREE ARMS, THREE JOBS.
  ours_gpu        site_radix, CANNS_RIPSER_STATS=1 -> our stage breakdown
  ripserpp_build3 the Oct 19 build, PROFILING enabled -> ripser++'s stage
                  breakdown (it prints per-phase timings to stderr)
  ripserpp_build2 the Oct 17 build, no instrumentation -> the HONEST baseline
                  total. build3 carries the cost of the instrumentation, so
                  the headline gap uses build2 and the stage split uses
                  build3. The difference between the two builds is measured
                  in the same batch, so it is visible rather than assumed.

TWO TOTALS, BECAUSE THEY ARE DIFFERENT THINGS.
  * "library total" -- the computation itself. Ours is PHASESTATS wall; theirs
    is ripser++'s own "total time". This is the ALGORITHMIC gap.
  * "process wall"  -- what a user waits for, measured with /usr/bin/time.
    Ours additionally pays Python interpreter startup and the numpy load of
    the input matrix; theirs is a C++ binary that reads a text file.
    This is the END-TO-END gap.
  Both are reported. The difference between them is not noise; it is the
  per-language startup tax, and it belongs in the report.

STAGE MAPPING. The two implementations do not have identical internal
boundaries, so the mapping is by FUNCTION, not by identical code path. The
mapping is written out in STAGES below and is printed with the output.

USAGE
    python gap_analysis.py --out raw.csv [--reps 5]
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
SITE = "/home/sichaohe/canns-lib-dev/site_flags"
# The same library built with `gpu-phase-stats`: the fine-grained timers
# inside the GPU apparent pre-pass (upload / alloc / kernel / D2H / host-side
# materialisation), printed as the GPUPHASE line. Without it the pre-pass
# is ONE opaque number, and the comparison with ripser++'s PROFILING lines
# is meaningless -- the two implementations cut their timers at different
# places, so "ours 3.850 vs theirs 0.045" was comparing a kernel against a
# whole block.
SITE_PHASE = "/home/sichaohe/canns-lib-dev/site_flags2"
RP3 = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"
RP2 = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build2/ripser++"
ASA_NPY = "/home/sichaohe/ph_work/ph_dm_1200.npy"
CAPPED_TXT = "/tmp/asa1200_capped_only.txt"
CHILD = "/home/sichaohe/ph_work/h2h_child.py"

# The stage vocabulary, shared by both sides.
STAGES = [
    "setup",       # reading/allocating before the filtration starts
    "edges",       # building + sorting the edge list
    "dim0",        # the H0 reduction
    "d1_scan",     # dim 1: the apparent-pair scan
    "d1_table",    # dim 1: building the pivot table
    "d1_reduce",   # dim 1: the reduction loop
    "d1_other",    # dim 1: the rest of the dimension
    "d2_scan",     # dim 2: the apparent-pair scan
    "d2_table",    # dim 2: building the pivot table
    "d2_reduce",   # dim 2: the reduction loop
    "d2_other",    # dim 2: the rest of the dimension
    "assemble",    # assembling the next dimension's columns
    "other",
    "unattributed",  # total minus the sum of the stages
]

# How each side's own output maps onto that vocabulary.
# ours: the STATS/PHASESTATS lines the library prints.
# ripser++: the PROFILING lines the Oct 19 build prints.
MAPPING_DOC = {
    "ours": {
        "setup": "not measured inside the library (the numpy load happens in the caller)",
        "edges": "PHASESTATS edges",
        "dim0": "PHASESTATS dim0",
        "d1_scan": "STATS dim=1 gpu_pre_s",
        "d1_table": "STATS dim=1 sharded_pivot_build_s",
        "d1_reduce": "STATS dim=1 loop_s",
        "d1_other": "STATS dim=1 total_s - (scan + table + reduce)",
        "d2_scan": "STATS dim=2 gpu_pre_s",
        "d2_table": "STATS dim=2 sharded_pivot_build_s",
        "d2_reduce": "STATS dim=2 loop_s",
        "d2_other": "STATS dim=2 total_s - (scan + table + reduce)",
        "assemble": "PHASESTATS assemble",
        "other": "PHASESTATS other",
    },
    "ripser++": {
        "setup": "'time to load distance matrix (I/O)' + 'CUDA PREPROCESSING TIME'",
        "edges": "not separately reported (folded into the dim 0 timing)",
        "dim0": "'0-dimensional persistence total computation time with GPU'",
        "d1_scan": "'gpu scan kernel time for dim: 1'",
        "d1_table": "'INSERTION POSTPROCESSING FOR GPU IN DIM 1'",
        "d1_reduce": "'SUBMATRIX REDUCTION TIME for dim 1'",
        "d1_other": "not separately reported",
        "d2_scan": "'gpu scan kernel time for dim: 2'",
        "d2_table": "'INSERTION POSTPROCESSING FOR GPU IN DIM 2'",
        "d2_reduce": "'SUBMATRIX REDUCTION TIME for dim 2'",
        "d2_other": "not separately reported",
        "assemble": "'time to copy hash map' + 'ASSEMBLE COLS TIME'",
        "other": "not separately reported",
    },
}


def num(s):
    m = re.search(r"([\d.]+)\s*s", s)
    return float(m.group(1)) if m else 0.0


def parse_ours(stderr):
    st = {}
    for line in stderr.splitlines():
        if line.startswith("PHASESTATS "):
            kv = dict(t.partition("=")[::2] for t in line.split()[1:] if "=" in t)
            st["wall"] = float(kv["wall"])
            st["edges"] = float(kv["edges"])
            st["dim0"] = float(kv["dim0"])
            st["assemble"] = float(kv["assemble"])
            st["other"] = float(kv["other"])
        elif line.startswith("STATS dim="):
            kv = dict(t.partition("=")[::2] for t in line.split()[1:] if "=" in t)
            d = kv["dim"]
            tot = float(kv["total_s"])
            scan = float(kv.get("gpu_pre_s", 0))
            table = float(kv.get("sharded_pivot_build_s", 0))
            red = float(kv.get("loop_s", 0))
            st["d%s_scan" % d] = scan
            st["d%s_table" % d] = table
            st["d%s_reduce" % d] = red
            st["d%s_other" % d] = tot - scan - table - red
    stages = {k: st.get(k, 0.0) for k in STAGES if k != "unattributed"}
    stages["setup"] = 0.0
    stages["edges"] = st.get("edges", 0.0)
    stages["dim0"] = st.get("dim0", 0.0)
    stages["assemble"] = st.get("assemble", 0.0)
    stages["other"] = st.get("other", 0.0)
    stages["unattributed"] = st.get("wall", 0.0) - sum(stages.values())
    return stages, st.get("wall", 0.0)


RP_RE = {
    "io": re.compile(r"time to load distance matrix \(I/O\):\s*([\d.]+)s"),
    "alloc": re.compile(r"CUDA PREPROCESSING TIME[^:]*:\s*([\d.]+)s"),
    "dim0": re.compile(r"0-dimensional persistence total computation time with GPU:\s*([\d.]+)s"),
    "total": re.compile(r"^total time:\s*([\d.]+)s", re.M),
    "scan1": re.compile(r"gpu scan kernel time for dim: 1:\s*([\d.]+)s"),
    "scan2": re.compile(r"gpu scan kernel time for dim: 2:\s*([\d.]+)s"),
    "ins1": re.compile(r"INSERTION POSTPROCESSING FOR GPU IN DIM 1:\s*([\d.]+)s"),
    "ins2": re.compile(r"INSERTION POSTPROCESSING FOR GPU IN DIM 2:\s*([\d.]+)s"),
    "hash2": re.compile(r"time to copy hash map for dim 2:\s*([\d.]+)s"),
    "asm2": re.compile(r"ASSEMBLE COLS TIME for dim 2:\s*([\d.]+)s"),
    "red1": re.compile(r"SUBMATRIX REDUCTION TIME for dim 1:\s*([\d.]+)s"),
    "red2": re.compile(r"SUBMATRIX REDUCTION TIME for dim 2:\s*([\d.]+)s"),
}


def parse_rp(stderr):
    g = {}
    for k, r in RP_RE.items():
        m = r.search(stderr)
        g[k] = float(m.group(1)) if m else 0.0
    stages = {
        "setup": g["io"] + g["alloc"],
        "edges": 0.0,
        "dim0": g["dim0"],
        "d1_scan": g["scan1"],
        "d1_table": g["ins1"],
        "d1_reduce": g["red1"],
        "d1_other": 0.0,
        "d2_scan": g["scan2"],
        "d2_table": g["ins2"],
        "d2_reduce": g["red2"],
        "d2_other": 0.0,
        "assemble": g["hash2"] + g["asm2"],
        "other": 0.0,
    }
    stages["unattributed"] = g["total"] - sum(stages.values())
    return stages, g["total"]


def parse_gpuphase(stderr):
    """The fine-grained timers of the GPU apparent pre-pass.

    The GPUPHASE line covers: upload (H2D), alloc, kernel, pin_read (D2H),
    build. It does NOT cover the host-side materialisation of the result
    into the consumer's type -- that happens AFTER the line is printed, so
    it is inferred as (the dimension's `gpu_pre_s` - the GPUPHASE total).

    The dimensions are identified by pairing each GPUPHASE line with the
    STATS line that follows it and checking that the column counts agree.
    """
    out = {}
    gpu_pre = {}
    for line in stderr.splitlines():
        if line.startswith("STATS dim="):
            kv = dict(t.partition("=")[::2] for t in line.split()[1:] if "=" in t)
            gpu_pre[kv["dim"]] = float(kv.get("gpu_pre_s", 0))
        elif line.startswith("GPUPHASE "):
            kv = dict(t.partition("=")[::2] for t in line.split()[1:] if "=" in t)
            cols = int(kv["cols"])
            out[cols] = {k: float(kv.get(k, 0))
                         for k in ("upload", "alloc", "kernel", "pin_read",
                                   "build", "total")}
    # Pair with the dimensions by column count.
    res = {}
    for line in stderr.splitlines():
        if line.startswith("STATS dim="):
            kv = dict(t.partition("=")[::2] for t in line.split()[1:] if "=" in t)
            d = kv["dim"]
            cols = int(kv["columns"])
            if cols in out:
                g = out[cols]
                g = dict(g)
                g["gpu_pre_s"] = gpu_pre.get(d, 0.0)
                g["unaccounted"] = gpu_pre.get(d, 0.0) - g["total"]
                res[d] = g
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    d = np.load(ASA_NPY)
    np.fill_diagonal(d, 0.0)
    cap = float(d[np.isfinite(d)].max()) * 2 + 1.0
    del d

    arms = ["ours_gpu", "ours_gpu_phase", "ripserpp_build3", "ripserpp_build2"]
    expected = len(arms) * args.reps
    print("EXPECTED_CELLS %d (arms=%d x reps=%d)" % (expected, len(arms), args.reps),
          flush=True)
    print("capped input, cap=%.6f, maxdim=2, coeff=47" % cap, flush=True)
    print("load before: %.2f" % float(open("/proc/loadavg").read().split()[0]),
          flush=True)

    rows = []
    for rep in range(1, args.reps + 1):
        for arm in arms:
            la = float(open("/proc/loadavg").read().split()[0])
            if arm == "ours_gpu" or arm == "ours_gpu_phase":
                site = SITE if arm == "ours_gpu" else SITE_PHASE
                env = dict(os.environ, PYTHONPATH=site, CANNS_RIPSER_STATS="1")
                cmd = ["/usr/bin/time", "-f", "__T__%e %M", PY, CHILD,
                       site, "capped", "2"]
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   env=env, timeout=args.timeout)
                stages, lib_total = parse_ours(p.stderr)
                fine = parse_gpuphase(p.stderr) if arm == "ours_gpu_phase" else {}
            elif arm == "ripserpp_build3":
                env = dict(os.environ)
                cmd = ["/usr/bin/time", "-f", "__T__%e %M", RP3, "--format",
                       "distance", "--dim", "2", "--threshold", "%.7g" % cap,
                       "--ratio", "1.0", CAPPED_TXT]
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   env=env, timeout=args.timeout)
                stages, lib_total = parse_rp(p.stderr)
            else:
                env = dict(os.environ)
                cmd = ["/usr/bin/time", "-f", "__T__%e %M", RP2, "--format",
                       "distance", "--dim", "2", "--threshold", "%.7g" % cap,
                       "--ratio", "1.0", CAPPED_TXT]
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   env=env, timeout=args.timeout)
                stages, lib_total = parse_rp(p.stderr)
                fine = {}
            m = re.search(r"__T__([\d.]+) (\d+)", p.stderr)
            wall = float(m.group(1)) if m else float("nan")
            rss = int(m.group(2)) if m else 0
            # build2 has NO instrumentation, so there is no stage breakdown for
            # it -- only a total. That has to be said explicitly in the CSV,
            # because a row of zeros that is silently averaged with a real
            # breakdown is how a "the gap is in stage X" claim gets made on
            # numbers that were never measured.
            has_bd = 0 if arm == "ripserpp_build2" else 1
            for s in STAGES:
                v = stages.get(s, 0.0)
                if not has_bd and s == "unattributed":
                    v = wall
                rows.append(dict(arm=arm, rep=rep, stage=s,
                                 seconds=round(v, 6), has_breakdown=has_bd))
            lt = lib_total if has_bd else wall
            rows.append(dict(arm=arm, rep=rep, stage="library_total",
                             seconds=round(lt, 6), has_breakdown=has_bd))
            rows.append(dict(arm=arm, rep=rep, stage="process_wall",
                             seconds=round(wall, 6), has_breakdown=has_bd))
            rows.append(dict(arm=arm, rep=rep, stage="peak_rss_kb",
                             seconds=rss, has_breakdown=has_bd))
            # Fine-grained rows for the instrumented arm. They are prefixed
            # `fine_` so the closure check (which sums the shared stage
            # vocabulary) does not see them: they are a SPLIT of an
            # existing stage, not extra work.
            for d, g in (fine or {}).items():
                for k, v in g.items():
                    rows.append(dict(arm=arm, rep=rep,
                                     stage="fine_d%s_%s" % (d, k),
                                     seconds=round(v, 6), has_breakdown=has_bd))
            print("  rep%d %-18s wall=%7.3f lib=%7.3f rss=%6.2fGB load=%.2f"
                  % (rep, arm, wall, lib_total, rss / 1048576.0, la), flush=True)

    fields = ["arm", "rep", "stage", "seconds", "has_breakdown"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows of %d expected cells)"
          % (args.out, len(rows), expected), flush=True)

    print("\n=== stage medians ===", flush=True)
    for arm in arms:
        print("  %s" % arm, flush=True)
        for s in STAGES + ["library_total", "process_wall"]:
            v = [r["seconds"] for r in rows
                 if r["arm"] == arm and r["stage"] == s]
            if v:
                print("    %-14s median=%8.3f  (n=%d)" % (s, statistics.median(v), len(v)),
                      flush=True)
    print("GAP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
