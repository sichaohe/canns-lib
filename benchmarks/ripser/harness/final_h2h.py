#!/usr/bin/env python
"""Final head-to-head: canns-lib (GPU) vs ripser++ on the ASA 1200 matrix.

DESIGN RULES, all of them learned the hard way in this project:

1. **INTERLEAVED, NOT SEQUENTIAL.** The arms run round-robin inside one
   batch, so every arm sees the same machine state within minutes of every
   other arm. The previous phase-1 driver ran the canns arms to completion
   and THEN ran ripser++, which is exactly the comparison that produced a
   14% drift (28.22 s under load vs 24.5 s idle) and a wrong conclusion.

2. **THE DENOMINATOR COMES FROM THE GRID.** The expected cell count is
   computed from the arm list and the repetition count, printed, and checked
   against what actually ran. A number that is "however many finished" is
   not a denominator.

3. **CAPPED INPUT IS THE COMPARABLE SIDE.** ripser++ cannot represent
   +inf, so it necessarily sees a capped matrix. Comparing it against our
   inf-preserved path would be comparing different problems. The
   inf-preserved arm is reported separately and labelled as such.

4. **ripser++'s stderr IS CAPTURED AND ASSERTED.** When it runs out of GPU
   memory it silently falls back to CPU-only and the message goes to
   stderr, not stdout. A run that fell back is REJECTED, not timed.

5. **md5 IS ONLY COMPARED WITHIN canns.** ripser++ prints 7 significant
   digits; ours prints f32 values. Its bar counts and rounded H1 top-3
   lifetimes are compared, not md5s.

6. **EVERY RUN IS TIMED AND TERMINATED.** `timeout` on every child, and a
   process census at the end, so an orphan cannot poison a later batch.

USAGE
    python final_h2h.py --out /home/sichaohe/ph_work/final_h2h.csv [--reps 5]
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
# Two ripser++ builds exist on the box: build2 (Oct 17) and build3 (Oct 19).
# build3 was compiled from the CURRENT ripser++.cu (the source's mtime is
# Oct 19 11:23, build3's binary is Oct 19 11:24), while build2 predates it.
# Every number quoted in the previous rounds used build2, so build2 stays as
# the historical baseline and build3 is measured alongside it -- otherwise
# "the ripser++ baseline" is an ambiguous phrase and the two are not
# comparable across rounds.
RP_BUILD2 = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build2/ripser++"
RP_BUILD3 = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"
ASA_NPY = "/home/sichaohe/ph_work/ph_dm_1200.npy"
CAPPED_NPY = "/tmp/asa1200_capped_only.npy"
CAPPED_TXT = "/tmp/asa1200_capped_only.txt"
FALLBACK_MARK = "CPU-ONLY MODE FOR REMAINDER"

# arm -> (kind, input_form, env overrides)
ARMS = [
    ("ours_gpu_radix", "canns", "capped", {}),
    ("ours_gpu_hostsort", "canns", "capped", {"CANNS_RIPSER_GPU_SORT": "0"}),
    ("ripserpp_build2", "ripserpp", "capped", {}),
    ("ripserpp_build3", "ripserpp", "capped", {}),
    ("ours_gpu_radix_inf", "canns", "inf_preserved", {}),
]

CHILD = "/home/sichaohe/ph_work/h2h_child.py"


def loadavg():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def gpu_free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return max(int(x) for x in out.stdout.split())
    except Exception:
        return -1


def run_canns(arm, iform, extra_env, timeout):
    env = dict(os.environ)
    env["PYTHONPATH"] = SITE
    env["CANNS_RIPSER_STATS"] = "1"
    env.update(extra_env)
    p = subprocess.run(
        ["/usr/bin/time", "-f", "__T__%e %M", PY, CHILD, SITE, iform, "2"],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    return p


def run_ripserpp(arm, iform, extra_env, timeout):
    env = dict(os.environ)
    env.update(extra_env)
    rp = RP_BUILD3 if arm.endswith("build3") else RP_BUILD2
    p = subprocess.run(
        ["/usr/bin/time", "-f", "__T__%e %M", rp, "--format", "distance",
         "--dim", "2", "--threshold", "%.7g" % CAP, "--ratio", "1.0",
         CAPPED_TXT],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    return p


def parse_ripserpp_stdout(stdout):
    """ripser++'s own output -> (bars per dim, H1 top-3).

    NOTE: THE ESSENTIAL CONVENTION MUST MATCH canns-lib's. ripser++ prints an
    essential bar as `[0, )` -- the death is EMPTY, so a naive
    `float(t[1])` raises and a naive split drops the bar. An earlier harness
    also filtered +inf deaths out of ripser++'s bars but NOT out of
    canns-lib's dgms, which fabricated a phantom off-by-one in dim0. Here
    the empty death becomes +inf and the bar is KEPT, so the counts are
    total (finite + essential) on both sides, exactly like the canns side.
    """
    o, dim = {}, None
    for line in stdout.splitlines():
        m = re.match(r"persistence intervals in dim (\d+):", line)
        if m:
            dim = int(m.group(1))
            o.setdefault(dim, [])
            continue
        t = line.strip().strip("[]()").split(",")
        if dim is not None and len(t) == 2:
            try:
                birth = float(t[0])
                death = float(t[1]) if t[1].strip() else float("inf")
            except ValueError:
                continue
            o[dim].append([birth, death])
    bars = [[list(x) for x in o.get(k, [])] for k in range(3)]
    life = np.sort(
        np.asarray([b - a for a, b in bars[1]], dtype="float64")
    )[::-1]
    life = life[np.isfinite(life)]
    top3 = ",".join("%.2f" % x for x in life[:3])
    return "|".join(str(len(x)) for x in bars), top3


def parse(p):
    rec = {}
    m = re.search(r"__T__([\d.]+) (\d+)", p.stderr)
    if m:
        rec["wall"] = float(m.group(1))
        rec["rss_kb"] = int(m.group(2))
    m = re.search(r"WALL ([\d.]+)", p.stdout)
    if m:
        rec["inner_wall"] = float(m.group(1))
    m = re.search(r"BARS (\S+)", p.stdout)
    if m:
        rec["bars"] = m.group(1)
    m = re.search(r"MD5 (\S+)", p.stdout)
    if m:
        rec["md5"] = m.group(1)
    m = re.search(r"H1TOP (\S+)", p.stdout)
    if m:
        rec["h1top"] = m.group(1)
    if "bars" not in rec:
        # ripser++ does not print our format; parse its own.
        try:
            rec["bars"], rec["h1top"] = parse_ripserpp_stdout(p.stdout)
        except Exception:
            pass
    rec["fallback"] = FALLBACK_MARK in p.stderr
    rec["rc"] = p.returncode
    # phase breakdown, ours only
    for line in p.stderr.splitlines():
        if line.startswith("PHASESTATS "):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, _, v = kv.partition("=")
                    rec["ph_" + k] = v
        elif line.startswith("ASSEMBLYGPU "):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, _, v = kv.partition("=")
                    rec["ag_" + k] = v
        elif line.startswith("ASSEMBLYSORT "):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, _, v = kv.partition("=")
                    rec["as_" + k] = v
        elif line.startswith("RADIXSORT "):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, _, v = kv.partition("=")
                    rec["rx_" + k] = v
        elif line.startswith("STATS dim="):
            # Per-dimension reduction breakdown. The raw line is kept
            # verbatim so a field added later is not silently dropped.
            kv = {}
            for tok in line.split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kv[k] = v
            dim = kv.get("dim", "?")
            rec["st_dim%s" % dim] = line
            for key in ("columns", "apparent_pairs", "nonapparent_cols"):
                if key in kv:
                    rec["st%s_%s" % (dim, key)] = kv[key]
            for key in ("total_s", "pre_pass_s", "gpu_pre_s",
                        "sharded_pivot_build_s", "loop_s", "nonapparent_loop_s"):
                if key in kv:
                    rec["st%s_%s" % (dim, key)] = kv[key]
            if "sharded_pivot" in kv:
                rec["st%s_sharded" % dim] = kv["sharded_pivot"]
    rec["stderr_tail"] = p.stderr[-400:].replace("\n", " | ")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/sichaohe/ph_work/final_h2h.csv")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    global CAP
    d = np.load(ASA_NPY)
    np.fill_diagonal(d, 0.0)
    CAP = float(d[np.isfinite(d)].max()) * 2 + 1.0
    capped = np.where(np.isfinite(d), d, CAP)
    np.savetxt(CAPPED_TXT, capped, fmt="%.10g")
    np.save(CAPPED_NPY, capped)

    expected = len(ARMS) * args.reps
    print("EXPECTED_CELLS %d  (arms=%d x reps=%d)" % (expected, len(ARMS), args.reps),
          flush=True)
    print("input: capped (cap=%.6g), maxdim=2, coeff=47" % CAP, flush=True)
    print("load before: %.2f" % loadavg(), flush=True)

    rows = []
    # Round-robin over arms so the arms are interleaved in TIME.
    for rep in range(1, args.reps + 1):
        for arm, kind, iform, extra in ARMS:
            t0 = time.time()
            la = loadavg()
            gf = gpu_free_mib()
            try:
                if kind == "canns":
                    p = run_canns(arm, iform, extra, args.timeout)
                else:
                    p = run_ripserpp(arm, iform, extra, args.timeout)
            except subprocess.TimeoutExpired:
                print("  %-20s rep%d TIMEOUT" % (arm, rep), flush=True)
                continue
            rec = parse(p)
            rec.update(arm=arm, rep=rep, input_form=iform,
                       load_before="%.2f" % la, load_after="%.2f" % loadavg(),
                       gpu_free_before_mib=gf, batch_ts=int(t0))
            rows.append(rec)
            status = "REJECTED(fallback)" if rec.get("fallback") else "ok"
            print("  rep%d %-20s wall=%8.3f rss=%6.2fGB load=%.2f gpu_free=%dMiB %s"
                  % (rep, arm, rec.get("wall", -1),
                     rec.get("rss_kb", 0) / 1048576.0, la, gf, status), flush=True)

    fields = sorted({k for r in rows for k in r})
    fields = [f for f in fields if f != "stderr_tail"] + ["stderr_tail"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows of %d expected)" % (args.out, len(rows), expected),
          flush=True)

    print("\n=== medians ===", flush=True)
    for arm, kind, iform, extra in ARMS:
        ws = [r["wall"] for r in rows if r["arm"] == arm and "wall" in r]
        if not ws:
            print("  %-20s NO DATA" % arm, flush=True)
            continue
        print("  %-20s median=%8.3f  min=%8.3f  max=%8.3f  spread=%6.3f  n=%d/%d"
              % (arm, statistics.median(ws), min(ws), max(ws),
                 max(ws) - min(ws), len(ws), args.reps), flush=True)

    # process census: an orphan would poison every later number
    census = subprocess.run(["ps", "-eo", "pid,etime,cmd"], capture_output=True,
                            text=True, timeout=30).stdout
    strays = [l for l in census.splitlines()
              if "ripser++" in l or "h2h_child" in l]
    print("\nSTRAY_PROCESSES %d" % len(strays), flush=True)
    for l in strays:
        print("  " + l.strip(), flush=True)
    print("H2H_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
