#!/usr/bin/env python3
"""The scale scan, with the fair comparison.

THE QUESTION THIS SCRIPT SETTLES. A reader can point at the first version
of this scan and object: ripser++ was run with `--threshold inf` on the
capped matrix, so the comparison was against the configuration that turns
every missing pair into a real edge. But ripser++ can be run with
`--threshold = max finite`, and then the capped pairs are excluded and the
filtration is the SAME as the one we get by keeping the `+inf`. That is the
configuration a ripser++ user would actually choose, so it is the one the
claim has to be made against.

THE FIVE ARMS, all on the same instance:

  ripserpp_all  ripser++ on the capped matrix, threshold = inf  (the CAP arm)
  ripserpp_T    ripser++ on the capped matrix, threshold = T    (the fair one)
  ours_inf      us, on the matrix with the `+inf` kept
  ours_T        us, on the capped matrix, threshold = T         (control)
  ours_all      us, on the capped matrix, threshold = inf       (the CAP arm)

`ours_inf` and `ours_T` compute the SAME filtration, so they must produce
the same bar count; `ours_T` is what the `+inf` representation saves us
inside our own engine. `ripserpp_T` is the honest cross-implementation
comparison. The bar counts of all three are compared, and the deaths are
compared for the bars that die at or below T (the ones above T are
truncated to infinity in the T arms, which is a property of the
representation, not of the implementation).

THE ORDER OF THE ARMS. Within a size, ours runs first and the reference
second. The reference's own allocation is the one that can fail (it asks
for more device memory than the card has), and a failed allocation leaves
the driver in a state that slows down whatever runs next -- measuring our
side after the reference has already failed would charge us for its
failure. Each program exits before the next one starts, so the device
memory itself is released; what is left is only the driver state, and the
reference cannot wedge it for us if it runs second.

THE WALL CLOCK. `wall_s` is the PROCESS wall clock of both sides: the same
convention on both, cold start included, so neither side is credited with
an initialization it did not have to pay. `lib_wall_s` is additionally
recorded for our side, which is the library's own clock, excluding the
interpreter start and the matrix load. The paper states both.

USAGE
    python scale_scan.py --site <site> --sizes 1200,1600,2000 [--reps 1]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time

import numpy as np

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
PP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"
SRC = "/home/sichaohe/canns-lib-dev"

CHILD = r'''
import json, sys, time
import numpy as np
site, path, maxdim, coeff, thresh = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                     int(sys.argv[4]), float(sys.argv[5]))
sys.path.insert(0, site)
from canns_lib.ripser import ripser
d = np.load(path)
t0 = time.perf_counter()
r = ripser(d, maxdim=maxdim, thresh=thresh, coeff=coeff,
           distance_matrix=True, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
bars = {str(d): [list(map(float, b)) for b in x] for d, x in enumerate(r)}
print("WALL %.4f" % wall)
print("BARS " + json.dumps(bars))
'''

FIELDS = ["n", "side", "threshold", "wall_s", "lib_wall_s", "rss_kb", "bars",
          "bars_d0", "bars_d1", "bars_d2", "error", "note", "loadavg",
          "threads", "commit", "build_md5"]


<<<<<<< HEAD
def make_instance(n, dim, missing_frac, seed, path):
    """A point cloud with a controlled fraction of missing pairs."""
=======
def run_timed(argv, stdout_path, timeout):
    """Run with a wall-clock timeout that ACTUALLY fires, and report how the
    program ended.

    `/usr/bin/time` is the parent of the program under test, so killing it
    leaves the program running as an orphan holding the stderr pipe open,
    and `subprocess.run(timeout=...)` then waits forever on the pipe. The
    child goes into its own process group and the GROUP is killed.

    stderr goes to a FILE, not a pipe: a program that is killed while it
    holds the pipe open would leave communicate() waiting forever, and the
    partial stderr -- which is where ripser++ says how much device memory
    it wanted -- would be lost with the pipe.

    Returns (stdout, stderr, returncode). The return code is not optional:
    a program that CRASHED has produced a file with no measurement in it,
    and a harness that records that as "no error" is reporting a failure as
    a result.
    """
    fo = open(stdout_path, "w")
    fe = open(stdout_path + ".err", "w")
    proc = subprocess.Popen(argv, stdout=fo, stderr=fe, text=True,
                            start_new_session=True)
    try:
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # The program may be stuck in an uninterruptible kernel call (a huge
        # CUDA allocation), in which case the kill stays pending and the
        # pipes do not close. Do not wait for it: the run is a TIMEOUT
        # either way, and the partial stderr is on disk.
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        fo.close()
        fe.close()
        raise
    fo.close()
    fe.close()
    err = open(stdout_path + ".err").read()
    return open(stdout_path).read(), err, proc.returncode


def make_instance(n, dim, missing_frac, seed, path_inf, path_capped, path_txt):
>>>>>>> d78f3da (fix(bench): the scale scan must fail loudly when an arm produces no number)
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, dim))
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 0.0)
<<<<<<< HEAD
    if missing_frac > 0:
        # Remove pairs at random, symmetrically: a pair is either present
        # for both vertices or for neither.
        iu = np.triu_indices(n, 1)
        keep = rng.random(iu[0].size) >= missing_frac
        mask = np.ones((n, n), dtype=bool)
        mask[iu[0][~keep], iu[1][~keep]] = False
        mask = mask | mask.T
        d = np.where(mask, d, np.inf)
    np.save(path, d)
    return d


def gpu_free_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1
=======
    iu = np.triu_indices(n, 1)
    keep = rng.random(iu[0].size) >= missing_frac
    mask = np.ones((n, n), dtype=bool)
    mask[iu[0][~keep], iu[1][~keep]] = False
    # AND, not OR: a pair is present only if BOTH directions were kept.
    mask = mask & mask.T
    d = np.where(mask, d, np.inf)
    np.save(path_inf, d)
    t = float(d[np.isfinite(d)].max())
    capped = np.where(np.isfinite(d), d, t * 2 + 1.0)
    np.save(path_capped, capped)
    np.savetxt(path_txt, capped, fmt="%.10g")
    return t, 1.0 - int((np.isfinite(d).sum() - n) // 2) / (n * (n - 1) // 2)


def md5_of(path):
    try:
        return hashlib.md5(open(path, "rb").read()).hexdigest()[:16]
    except OSError:
        return ""


def extension_md5(site):
    """The md5 of the built extension. It is inside the package directory, so
    the search is recursive -- a top-level-only search finds nothing."""
    for root, _dirs, names in os.walk(site):
        for name in sorted(names):
            if name.endswith(".so"):
                return md5_of(os.path.join(root, name))
    return ""


def source_commit():
    """The fingerprint of the source the build came from.

    The tree on the benchmark host is a copy, not a git checkout, so the
    commit comes from git when there is one and otherwise from the md5 of
    the files the build actually depends on.
    """
    try:
        c = subprocess.run(["git", "-C", SRC, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        if c:
            return c
    except Exception:
        pass
    h = hashlib.md5()
    for rel in ("src/ripser/core/reduction.rs", "src/ripser/matrix/sparse.rs",
                "src/ripser/core/algorithm.rs", "Cargo.toml"):
        try:
            h.update(open(os.path.join(SRC, rel), "rb").read())
        except OSError:
            pass
    return "src:%s" % h.hexdigest()[:12]


def reference_note(stderr_text):
    """What the reference itself says about the run.

    A timeout is only a RESULT if it says WHY, and so is a run that fell
    back to the host: ripser++ says how much device memory it wanted, and
    whether it gave up on the GPU.
    """
    bits = []
    m = re.search(r"gpu memory needed for full rips by calculation in "
                  r"bytes for dim: \d+: (\d+)", stderr_text)
    if m:
        bits.append("needs %.1f GB of device memory" % (int(m.group(1)) / 1e9))
    m = re.search(r"recalculated dim_max based on GPU free DRAM capacity: "
                  r"(\d+)", stderr_text)
    if m:
        bits.append("recalculated dim_max to %s" % m.group(1))
    if "CPU-ONLY MODE FOR REMAINDER" in stderr_text:
        bits.append("FELL BACK TO CPU-ONLY (not enough device memory)")
    return "; ".join(bits)


def parse_ours(out_text):
    """(library wall, bars per dimension) from our own output."""
    wall, bars = None, None
    for line in out_text.splitlines():
        if line.startswith("WALL "):
            wall = float(line.split()[1])
        elif line.startswith("BARS "):
            bars = json.loads(line[5:])
    if not isinstance(bars, dict):
        bars = {}
    return wall, bars


def parse_reference(out_text):
    """(total bars, bars per dimension) from the reference's output."""
    per, dim = {}, None
    for ln in out_text.splitlines():
        mm = re.match(r"persistence intervals in dim (\d+):", ln)
        if mm:
            dim = int(mm.group(1))
            per.setdefault(dim, 0)
        elif dim is not None and ln.startswith(" ["):
            per[dim] += 1
    return sum(per.values()), per


def write_table(path, rows):
    """Write the table. Called after every size, so an interrupted run keeps
    what it has already measured."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
>>>>>>> d78f3da (fix(bench): the scale scan must fail loudly when an arm produces no number)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/home/sichaohe/canns-lib-dev/site_ship6")
    ap.add_argument("--sizes", default="1200,1600,2000")
    ap.add_argument("--dim", type=int, default=9)
    ap.add_argument("--missing", type=float, default=0.30)
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--coeff", type=int, default=47)
<<<<<<< HEAD
    ap.add_argument("--reps", type=int, default=2)
=======
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-run wall-clock limit; a run that hits it is "
                         "recorded as a TIMEOUT, which is a result")
>>>>>>> d78f3da (fix(bench): the scale scan must fail loudly when an arm produces no number)
    ap.add_argument("--out", default="scale_scan.csv")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names to run; empty = all five. "
                         "A side can be measured in its own pass (the "
                         "reference's arms are timeouts, which are expensive "
                         "to reproduce) and the tables merged.")
    args = ap.parse_args()
    arm_filter = {a.strip() for a in args.arms.split(",") if a.strip()}

    commit = source_commit()
    our_md5 = extension_md5(args.site)
    pp_md5 = md5_of(PP)
    threads = len(os.sched_getaffinity(0))
    print("commit=%s  our_md5=%s  ref_md5=%s  threads=%d"
          % (commit, our_md5, pp_md5, threads), flush=True)

    rows = []
    for n in [int(x) for x in args.sizes.split(",") if x.strip()]:
        p_inf = "/tmp/_scale_%d_inf.npy" % n
        p_cap = "/tmp/_scale_%d_cap.npy" % n
        p_txt = "/tmp/_scale_%d.txt" % n
        t, inf_frac = make_instance(n, args.dim, args.missing, 1234 + n,
                                    p_inf, p_cap, p_txt)
        print("=== n=%d ===  missing pairs %.4f  T=%.6g" % (n, inf_frac, t),
              flush=True)

<<<<<<< HEAD
        # --- ripser++ (capped, the only form it can read) ------------------
        t0 = time.perf_counter()
        txt = "/tmp/_scale_%d.txt" % n
        np.savetxt(txt, np.where(fin, d, float(d[fin].max()) * 2 + 1.0),
                   fmt="%.10g")
        txt_write = time.perf_counter() - t0
        pp_best, pp_rss, pp_bars, pp_err = None, 0, None, ""
        for rep in range(args.reps):
            free0 = gpu_free_mib()
            try:
                with open("/tmp/_scale_pp.out", "w") as fo:
                    t0 = time.perf_counter()
                    p = subprocess.run(["/usr/bin/time", "-f", "__T__%e %M", PP,
                                        "--format", "distance", "--dim",
                                        str(args.maxdim), "--threshold", "inf",
                                        "--ratio", "1.0", txt],
                                       stdout=fo, stderr=subprocess.PIPE,
                                       text=True, timeout=3600)
                    wall = time.perf_counter() - t0
                m = re.search(r"__T__([\d.]+) (\d+)", p.stderr)
                rss = int(m.group(2)) if m else 0
                if pp_best is None or wall < pp_best:
                    pp_best, pp_rss = wall, rss
                pp_bars = sum(1 for ln in open("/tmp/_scale_pp.out")
                              if ln.startswith(" ["))
                if "CUDA error" in p.stderr or "out of memory" in p.stderr.lower():
                    pp_err = "GPU_OOM"
            except subprocess.TimeoutExpired:
                pp_err = "TIMEOUT"
        print("  ripser++: %s  rss=%s MB  bars=%s %s"
              % ("%.3f s" % pp_best if pp_best else "FAILED",
                 pp_rss // 1024, pp_bars, pp_err), flush=True)
        rows.append({"n": n, "side": "ripserpp", "wall_s": pp_best or "",
                     "rss_kb": pp_rss, "gpu_free_mib_before": free0,
                     "bars": pp_bars or "", "md5": "", "error": pp_err,
                     "loadavg": open("/proc/loadavg").read().split()[0]})
=======
        # ours first, the reference second -- see the docstring.
        plan = [("ours_inf", "ours", p_inf, "inf"),
                ("ours_T", "ours", p_cap, "%.10g" % t),
                ("ours_all", "ours", p_cap, "inf"),
                ("ripserpp_T", "ripserpp", p_txt, "%.10g" % t),
                ("ripserpp_all", "ripserpp", p_txt, "inf")]
>>>>>>> d78f3da (fix(bench): the scale scan must fail loudly when an arm produces no number)

        for name, side, path, thresh in plan:
            if arm_filter and name not in arm_filter:
                continue
            best, rss, bars, err, note = None, 0, None, "", ""
            lib_wall = None
            for rep in range(args.reps):
                if side == "ours":
                    argv = ["/usr/bin/time", "-f", "__T__%e %M", PY, "-c",
                            CHILD, args.site, path, str(args.maxdim),
                            str(args.coeff), thresh]
                    out_path = "/tmp/_scale_ours.out"
                else:
                    argv = ["/usr/bin/time", "-f", "__T__%e %M", PP,
                            "--format", "distance", "--dim", str(args.maxdim),
                            "--threshold", thresh, "--ratio", "1.0", path]
                    out_path = "/tmp/_scale_pp.out"
                t0 = time.perf_counter()
                try:
                    out_text, err_text, rc = run_timed(argv, out_path,
                                                        args.timeout)
                except subprocess.TimeoutExpired:
                    err = "TIMEOUT"
                    note = "did not finish inside %d s" % args.timeout
                    # the reference's own words survive the kill, on disk
                    try:
                        note = "; ".join(
                            x for x in [note, reference_note(
                                open(out_path + ".err").read())] if x)
                    except OSError:
                        pass
                    break
                wall = time.perf_counter() - t0
                if rc != 0:
                    # A crash is not a measurement. Record it as a failure,
                    # with the end of the child's stderr as the evidence --
                    # a crash that produced no number used to be recorded
                    # as a row with no number and no error.
                    err = "CRASH"
                    note = ("exited with status %d" % rc) + (
                        "; " + err_text.strip().splitlines()[-1]
                        if err_text.strip() else "")
                    break
                m = re.search(r"__T__([\d.]+) (\d+)", err_text)
                r = int(m.group(2)) if m else 0
                if side == "ours":
                    lw, b = parse_ours(out_text)
                    if lw is None:
                        err = "NO_MEASUREMENT"
                        note = "the child produced no WALL line"
                        break
                    lib_wall = lw
                    total = sum(len(v) for v in b.values())
                    per = {d: len(v) for d, v in b.items()}
                else:
                    total, per = parse_reference(out_text)
                    note = reference_note(err_text)
                    if "CUDA error" in err_text or "out of memory" in err_text.lower():
                        err = "GPU_OOM"
                if best is None or wall < best:
                    best, rss = wall, r
<<<<<<< HEAD
                for line in open("/tmp/_scale_ours.out"):
                    if line.startswith("BARS "):
                        bars = json.loads(line[5:])
            md5 = ""
            if bars:
                md5 = hashlib.md5(
                    json.dumps({k: sorted(map(tuple, v)) for k, v in bars.items()},
                               sort_keys=True).encode()).hexdigest()[:16]
            print("  ours %-7s %8.3f s  rss=%6d MB  bars=%s"
                  % (form, best or 0, rss // 1024,
                     sum(len(v) for v in (bars or {}).values())), flush=True)
            rows.append({"n": n, "side": "ours_" + form, "wall_s": best or "",
                         "rss_kb": rss, "gpu_free_mib_before": -1,
                         "bars": sum(len(v) for v in (bars or {}).values()),
                         "md5": md5, "error": "",
                         "loadavg": open("/proc/loadavg").read().split()[0]})
        os.unlink(path)
        os.unlink(txt)
=======
                bars = (total, per)
            if isinstance(bars, tuple):
                total, per = bars
            else:
                total, per = "", {}
            rows.append({"n": n, "side": name,
                         "threshold": ("inf" if thresh == "inf" else "T"),
                         "wall_s": "%.4f" % best if best else "",
                         "lib_wall_s": "%.4f" % lib_wall if lib_wall else "",
                         "rss_kb": rss, "bars": total,
                         "bars_d0": per.get(0, per.get("0", "")),
                         "bars_d1": per.get(1, per.get("1", "")),
                         "bars_d2": per.get(2, per.get("2", "")),
                         "error": err, "note": note,
                         "loadavg": open("/proc/loadavg").read().split()[0],
                         "threads": threads if side == "ours" else "",
                         "commit": commit if side == "ours" else "",
                         "build_md5": (our_md5 if side == "ours" else pp_md5)})
            print("  %-14s %8.2f s  lib=%8.2f s  rss=%6d MB  bars=%s %s %s"
                  % (name, best or 0, lib_wall or 0, rss // 1024, total,
                     err, note), flush=True)
        for f in (p_inf, p_cap, p_txt):
            if os.path.exists(f):
                os.unlink(f)
        write_table(args.out, rows)
>>>>>>> d78f3da (fix(bench): the scale scan must fail loudly when an arm produces no number)

    fields = ["n", "side", "wall_s", "rss_kb", "gpu_free_mib_before", "bars",
              "md5", "error", "loadavg"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
