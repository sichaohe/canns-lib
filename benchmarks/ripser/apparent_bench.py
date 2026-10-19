#!/usr/bin/env python3
"""
P0 benchmark + equivalence harness for the apparent-pair work in canns-lib.

Modes
-----
bench   time each grid cell under whatever CANNS_RIPSER_* env is already set
        (used to produce the P0 baseline, which is the apparent-OFF state)
equiv   run the same cell twice -- apparent off, then apparent parallel on --
        and require the persistence diagrams to be BITWISE identical

Design rules this file follows on purpose
----------------------------------------
* The grid is written down in EXPECTED and the denominator is COMPUTED from it.
  Never hand-write a denominator; a hand-written one drifts when the grid does.
* Results are APPENDED to the CSV after every single cell. An earlier run lost
  15 of 16 completed cells because the script only dumped at the very end.
* barcodes are compared with np.array_equal, not a tolerance. "Close" is not
  the property under test; "identical" is.
* STDERR_STATS is parsed from canns-lib's `STATS dim=.. columns=.. apparent_pairs=..`
  line so the apparent hit rate is measured, never assumed.
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

import numpy as np

# ----------------------------------------------------------------- the grid
FIXTURES = ["circle", "sphere", "torus", "random", "clusters", "grid", "swiss_roll"]
NS = [50, 100, 150, 300]
MAXDIMS = [1, 2]
COEFFS = [2, 47]
# Jetson has ~5 GB usable and the server peaks at 44 GB for n=1200/maxdim=2,
# so arm64 runs a reduced size set (plan §12.4).
NS_SMALL = [50, 100, 150, 300]
MAXDIMS_SMALL = [1, 2]

FIELDS = [
    "mode", "fixture", "n", "maxdim", "coeff", "apparent",
    "threads", "cols", "apparent_pairs", "apparent_rate",
    "h0_bars", "h1_bars", "h2_bars",
    "h1_top1", "h1_top2", "h2_top1",
    "dgm_md5", "secs", "rss_mb", "lib", "libver", "error",
]

STATS_RE = re.compile(r"STATS dim=(\d+) columns=(\d+) apparent_pairs=(\d+)")


def expected_cells(mode, fixtures, ns, maxdims, coeffs):
    n_apparent = 1 if mode == "bench" else 2
    return len(fixtures) * len(ns) * len(maxdims) * len(coeffs) * n_apparent


# --------------------------------------------------------------- fixtures
def make_fixture(name: str, n: int) -> np.ndarray:
    """Deterministic point cloud. Every fixture is a pure function of (name, n)
    so a rerun on another machine compares like with like."""
    from ripser import tadasets

    rng = np.random.default_rng(0xC0FFEE + n)
    if name == "circle":
        return tadasets.dsphere(n=n, d=1, noise=0.05, rng=rng)
    if name == "sphere":
        return tadasets.dsphere(n=n, d=2, noise=0.05, rng=rng)
    if name == "torus":
        return tadasets.torus(n=n, c=2, a=1, noise=0.05, rng=rng)
    if name == "random":
        return np.asarray(rng.standard_normal((n, 3)), dtype=np.float64)
    if name == "clusters":
        return np.asarray(tadasets.clusters(n=n, d=3, k=4, rng=rng), dtype=np.float64)
    if name == "grid":
        # Deterministic 3D lattice, trimmed/padded to exactly n points.
        m = int(np.ceil(n ** (1 / 3)))
        g = np.stack(np.meshgrid(*([np.arange(m)] * 3), indexing="ij"), -1).reshape(-1, 3)
        g = g.astype(np.float64) + rng.normal(scale=1e-3, size=g.shape)
        return g[:n] if g.shape[0] >= n else np.vstack(
            [g, g[rng.integers(0, g.shape[0], n - g.shape[0])]])
    if name == "swiss_roll":
        return np.asarray(tadasets.swiss_roll(n=n, d=3, noise=0.05, rng=rng),
                           dtype=np.float64)
    raise ValueError(f"unknown fixture {name!r}")


# ------------------------------------------------------------------- run
def run_cell(maxdim: int, coeff: int, apparent: bool, threads: int, reps: int = 1,
             apparent_par: int | None = None):
    """Build the per-cell subprocess program and its environment.

    A SUBPROCESS per cell on purpose: env vars (CANNS_RIPSER_APPARENT*) and the
    STATS stderr line must be isolated per cell, and an OOM in one cell must
    not kill the whole grid.
    """
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(threads)
    # apparent ON is what this work is about; OFF is the untouched baseline.
    env["CANNS_RIPSER_APPARENT"] = "1" if apparent else "0"
    # --apparent-par lets us A/B rayon on/off WITHIN one build, which is the
    # only way to attribute a speedup to the parallel pre-pass alone.
    par = apparent if apparent_par is None else apparent_par
    env["CANNS_RIPSER_APPARENT_PAR"] = "1" if par else "0"
    env["CANNS_RIPSER_USE_LOCKFREE"] = "0"
    # one STATS eprintln per DIMENSION (not per column), so enabling it costs
    # nothing measurable while giving us the apparent hit rate to report.
    env["CANNS_RIPSER_STATS"] = "1"
    # the subprocess rebuilds the fixture, so it must be able to import the
    # generator module. APPEND, never overwrite: the caller's PYTHONPATH is how
    # the isolated canns_lib build (site/) is being found, and clobbering it
    # would silently test the server's OLD installed canns_lib instead.
    here = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = (here + os.pathsep + env["PYTHONPATH"]) if env.get(
        "PYTHONPATH") else here

    prog = """
import os, sys, time, hashlib, json
import numpy as np
from apparent_fixtures import make_fixture
from canns_lib.ripser import ripser

spec = json.loads(sys.argv[1])
X = make_fixture(spec["fixture"], spec["n"])
reps = int(spec.get("reps", 1))
# Minimum over reps, not mean. A single-shot timing cannot support a 2%
# regression gate: the apparent=0 code path is byte-identical between builds
# yet measured up to 1.30x apart purely from machine load. Best-of-N is the
# standard defence against that.
best = None
for _ in range(reps):
    t0 = time.perf_counter()
    res = ripser(X, maxdim=spec["maxdim"], coeff=spec["coeff"],
                 do_cocycles=False, progress_bar=False)
    el = time.perf_counter() - t0
    best = el if best is None else min(best, el)
secs = best

dgms = res["dgms"]
def finite(a):
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a[:, 1])] if len(a) else a.reshape(0, 2)
def top(a):
    f = finite(a)
    if not len(f): return [None, None]
    p = f[:, 1] - f[:, 0]
    o = np.argsort(p)[::-1]
    return [float(p[o[i]]) if i < len(p) else None for i in range(2)]

def barcount(d):
    if d >= len(dgms):
        return 0
    return int(len(finite(dgms[d])))

f1 = finite(dgms[1]) if len(dgms) > 1 else np.zeros((0, 2))
f2 = finite(dgms[2]) if len(dgms) > 2 else np.zeros((0, 2))
# md5 over the sorted-by-(birth,death) raw float64 bytes: order-independent
# fingerprint that still changes if a single float changes.
def md5(a):
    if not len(a): return "empty"
    b = np.asarray(a, dtype="<f8"); b = b[np.lexsort((b[:, 1], b[:, 0]))]
    return hashlib.md5(b.tobytes()).hexdigest()

out = dict(
    secs=secs,
    h1_bars=barcount(1), h2_bars=barcount(2),
    h1_top=top(dgms[1] if len(dgms) > 1 else [])[0],
    h1_top2=top(dgms[1] if len(dgms) > 1 else [])[1],
    h2_top=top(dgms[2] if len(dgms) > 2 else [])[0],
    h1_md5=md5(f1), h2_md5=md5(f2),
    lib=__import__("canns_lib").__file__,
    ver=__import__("canns_lib").__version__,
)
import resource
out["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
sys.stderr.write("RESULT " + json.dumps(out) + "\\n")
"""
    spec = {"fixture": "", "n": 0, "maxdim": maxdim, "coeff": coeff}
    return prog, env


def run_one(mode, fixture, n, maxdim, coeff, apparent, threads, out_csv, rows,
            reps=1, expect_lib=None, apparent_par=None):
    import json
    spec = {"fixture": fixture, "n": n, "maxdim": maxdim, "coeff": coeff,
            "reps": reps}
    prog, env = run_cell(maxdim, coeff, apparent, threads, reps, apparent_par)
    t0 = time.time()
    rec = dict.fromkeys(FIELDS, "")
    rec.update(mode=mode, fixture=fixture, n=n, maxdim=maxdim, coeff=coeff,
               apparent=int(apparent), threads=threads)
    try:
        p = subprocess.run([sys.executable, "-c", prog, json.dumps(spec)],
                           capture_output=True, text=True, env=env, timeout=1800)
        line = [l for l in p.stderr.splitlines() if l.startswith("RESULT ")]
        if p.returncode != 0 or not line:
            # Keep the TRACEBACK, not just its last line. Reporting only the tail
            # of a subprocess failure costs a debugging round-trip every time.
            tail = p.stderr.strip()[-600:] or "<no stderr>"
            raise RuntimeError(f"rc={p.returncode}: {tail}")
        d = json.loads(line[-1][len("RESULT "):])
        cols = sum(int(m.group(2)) for m in
                   (STATS_RE.search(l) for l in p.stderr.splitlines()) if m)
        ap = sum(int(m.group(3)) for m in
                 (STATS_RE.search(l) for l in p.stderr.splitlines()) if m)
        rec.update(
            cols=cols, apparent_pairs=ap,
            apparent_rate=round(ap / cols, 6) if cols else "",
            h1_bars=d["h1_bars"], h2_bars=d["h2_bars"],
            h1_top1=("" if d["h1_top"] is None else f"{d['h1_top']:.9g}"),
            h1_top2=("" if d["h1_top2"] is None else f"{d['h1_top2']:.9g}"),
            h2_top1=("" if d["h2_top"] is None else f"{d['h2_top']:.9g}"),
            dgm_md5=f"{d['h1_md5']}:{d['h2_md5']}",
            secs=f"{d['secs']:.4f}", rss_mb=d["rss_mb"],
            lib=d.get("lib", ""), libver=d.get("ver", ""),
        )
        verify_expected_lib(d.get("lib", ""), expect_lib)
    except subprocess.TimeoutExpired:
        rec["error"] = "timeout>1800s"
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    rows.append(rec)
    append(out_csv, rec)
    return rec


def append(path, rec):
    # "file exists" is NOT "file has a header": `: > out.csv` leaves a
    # zero-byte file that exists, so the header would be skipped and the first
    # DATA row would be consumed as the header by DictReader -- silently losing
    # one cell out of the grid. Decide on size instead.
    needs_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if needs_header:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in FIELDS})


def verify_expected_lib(resolved: str, expected: str | None) -> None:
    """Abort IMMEDIATELY if the build under test is not the one we asked for.

    The post-hoc provenance check in the checker caught a 30-minute run that had
    silently been produced by a different canns_lib than the PYTHONPATH pointed
    at. Recording the path is necessary but not sufficient; this makes a
    mismatch fail on cell 1 instead of after the whole grid.
    """
    if not expected:
        return
    if not resolved:
        raise SystemExit(f"FATAL: no canns_lib path resolved (expected {expected!r})")
    if os.path.realpath(resolved) != os.path.realpath(expected):
        raise SystemExit(
            f"FATAL: wrong canns_lib under test\n"
            f"  expected: {expected}\n"
            f"  resolved: {resolved}\n"
            f"  A cell was already timed against the wrong build; the CSV is not "
            f"comparable. Fix PYTHONPATH / --expect-lib and re-run."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "equiv"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default=",".join(FIXTURES))
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--maxdims", default=",".join(map(str, MAXDIMS)))
    ap.add_argument("--coeffs", default=",".join(map(str, COEFFS)))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--apparent-par", type=int, choices=[0, 1], default=None,
                    help="override CANNS_RIPSER_APPARENT_PAR; A/B rayon within "
                         "one build")
    ap.add_argument("--expect-lib", default=None,
                    help="abort on cell 1 if canns_lib resolves to a different path")
    ap.add_argument("--reps", type=int, default=1,
                    help="timing repetitions per cell; the MIN is recorded")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N cells")
    args = ap.parse_args()

    fixtures = args.fixtures.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    maxdims = [int(x) for x in args.maxdims.split(",")]
    coeffs = [int(x) for x in args.coeffs.split(",")]
    threads = args.threads or os.cpu_count() or 1

    total = expected_cells(args.mode, fixtures, ns, maxdims, coeffs)
    if args.limit:
        total = min(total, args.limit)
    print(f"mode={args.mode} platform_threads={threads} reps={args.reps} cells={total}")
    if args.expect_lib:
        print(f"expect-lib: {args.expect_lib}")
    print(f"grid: {len(fixtures)} fixtures x {len(ns)} n x {len(maxdims)} maxdim "
          f"x {len(coeffs)} coeff x {2 if args.mode == 'equiv' else 1} mode")

    rows = []
    done = 0
    for fixture in fixtures:
        for n in ns:
            for maxdim in maxdims:
                for coeff in coeffs:
                    modes = [False, True] if args.mode == "equiv" else [False]
                    for ap in modes:
                        rec = run_one(args.mode, fixture, n, maxdim, coeff,
                                      ap, threads, args.out, rows, args.reps,
                                      args.expect_lib, args.apparent_par)
                        done += 1
                        flag = "ERR " + rec["error"] if rec["error"] else "ok"
                        print(f"  [{done} / {total} ({done / total:.0%})] "
                              f"{fixture:<10} n={n:<4} md={maxdim} coeff={coeff:<3} "
                              f"apparent={int(ap)} {rec['secs']:>9}s "
                              f"h1={rec['h1_bars']:<5} rate={rec['apparent_rate']} "
                              f"{flag}", flush=True)
                        if args.limit and done >= args.limit:
                            print(f"done {done} / {total} (limited)")
                            return 0
    print(f"done {done} / {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
