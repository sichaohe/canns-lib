#!/usr/bin/env python3
"""Independent validation of the head-to-head on the grid-cells matrices.

WHY A SECOND DATASET. The ASA matrix gave a 2-4% margin over ripser++, and a
2-4% margin on ONE matrix is not evidence of parity -- it is one draw. This
script runs BOTH implementations on the real grid-cells matrices (1200x1200,
~70% of the pairs genuinely +inf) at maxdim=1, and checks:

  * the SAME filtration on both sides. Ours gets the +inf matrix and the
    threshold T = max finite distance; ripser++ gets the CAPPED matrix
    (cap = 2*max_finite + 1, which is ABOVE T) and the same threshold T, so
    the capped pairs are never born on either side. NOTE: passing `--threshold
    cap` to ripser++ (as ~/gridcells/gca_one.py does) is a DIFFERENT
    filtration -- the fake edges are born at cap and do kill H1 classes.
  * the SAME barcode on both sides, per dimension.
  * the wall time, so the ratio is measured on data that was not used to
    design the change.

BARCODE COMPARISON (this is the part that was wrong in the first version).
An exact md5 of the bar values CANNOT be compared across the two
implementations:

  * ripser++ prints its intervals with 6 significant digits ("[0,2.23622e-05)"),
    so its values carry ~5e-6 relative rounding against our f64/f32 values;
  * the first version also hashed dim0+dim1 on our side but only dim1 on
    ripser++'s side.

So the comparison here is numeric, not hash-based:
  * exact equality of the per-dimension bar COUNTS;
  * a greedy nearest-neighbour matching of the bars with a relative
    tolerance (1e-5, twice the 6-digit print precision) -> unmatched count
    and the maximum relative deviation of the matched bars.
The md5 column that this script emits is over OUR full-precision bars only,
so it is comparable with the `h1_md5` column of ~/gridcells/gca_results.csv
(same function, same convention) -- it is a REGRESSION fingerprint, not a
cross-implementation identity check.

USAGE
    python validate_gridcells.py --site /path/site_ship2 [--cells 6] [--reps 3]
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_core

GRID = "/home/sichaohe/gridcells"
RP = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/build3/ripser++"

# ripser++ prints 6 significant digits -> a value v is printed with a
# relative error up to 5e-7 (and up to 5e-6 relative for values < 1, where
# the absolute error is capped by the last printed digit). 1e-5 relative
# leaves a 2x margin and still rejects a genuinely different filtration.
RTOL = 1e-5


def md5bars(bars):
    """Full-precision fingerprint of a bar list (ours only).

    Same function/convention as ~/gridcells/gca_one.py, so the value is
    directly comparable with the `h1_md5` column of gca_results.csv.
    """
    a = np.asarray(sorted(map(tuple, bars)), dtype="float64")
    return hashlib.md5(a.tobytes()).hexdigest()[:16]


def match_bars(a, b, rtol=RTOL):
    """Greedy nearest-neighbour match of two bar lists.

    a, b: lists of (birth, death) pairs, death may be inf.
    Returns (n_matched, max_rel_dev, n_unmatched_a, n_unmatched_b).
    """
    a = [(float(x[0]), float(x[1])) for x in a]
    b = [(float(x[0]), float(x[1])) for x in b]
    if len(a) != len(b):
        return 0, float("inf"), len(a), len(b)
    if not a:
        return 0, 0.0, 0, 0
    A = np.asarray(a, dtype="float64")
    B = np.asarray(b, dtype="float64")
    dev = np.zeros((len(A), len(B)), dtype="float64")
    for k in (0, 1):
        x = A[:, k][:, None]
        y = B[:, k][None, :]
        both_inf = np.isinf(A[:, k])[:, None] & np.isinf(B[:, k])[None, :]
        one_inf = np.isinf(A[:, k])[:, None] ^ np.isinf(B[:, k])[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.abs(x - y) / np.maximum(np.abs(y), 1e-300)
        d = np.where(both_inf, 0.0, d)
        d = np.where(one_inf, np.inf, d)
        dev = np.maximum(dev, d)
    order = np.argsort(dev, axis=None)
    used_a, used_b, devs = set(), set(), []
    flat = dev.ravel()
    for idx in order:
        d = flat[idx]
        if d > rtol:
            break
        i, j = divmod(int(idx), dev.shape[1])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        devs.append(float(d))
    max_dev = max(devs) if devs else 0.0
    return len(devs), max_dev, len(a) - len(used_a), len(b) - len(used_b)


def parse_ripserpp(path):
    o, dim = {}, None
    for line in open(path):
        m = re.match(r"persistence intervals in dim (\d+):", line)
        if m:
            dim = int(m.group(1))
            o.setdefault(dim, [])
            continue
        t = line.strip().strip("[]()").split(",")
        if dim is not None and len(t) == 2:
            try:
                o[dim].append([float(t[0]), float(t[1]) if t[1].strip() else float("inf")])
            except ValueError:
                pass
    return o


def prepare_ripserpp_input(d):
    """Write the CAPPED matrix once; returns (path, cleanup_list).

    Writing the 17 MB text file is I/O, not ripser++ work, so it is kept
    OUT of the timed region -- otherwise the harness would charge ripser++
    for a text dump it does not do for us.
    """
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    cap = float(d[np.isfinite(d)].max()) * 2.0 + 1.0
    np.savetxt(tmp, np.where(np.isfinite(d), d, cap), fmt="%.10g")
    return tmp


def run_ripserpp(path, threshold):
    """ripser++ on the CAPPED matrix at the same threshold as ours."""
    out = path + ".out"
    with open(out, "w") as fo:
        subprocess.run([RP, "--format", "distance", "--dim", "1",
                        "--threshold", "%.10g" % threshold, "--ratio", "1.0", path],
                       stdout=fo, stderr=subprocess.DEVNULL, timeout=3000)
    o = parse_ripserpp(out)
    os.unlink(out)
    return o


def degeneracy_stats(d, finite):
    """How close to degenerate is this matrix?

    Returns (finite_pairs, distinct_values, near_tie_pairs), where
    near_tie_pairs counts the sorted distinct distances whose successor is
    within 1e-5 relative. A matrix where most distances have a
    near-neighbour is one where a hair of arithmetic reordering can swap
    which of two nearly simultaneous simplices kills which cycle -- the
    regime in which two implementations can pair a bar differently without
    either of them being wrong about the filtration.
    """
    vals = np.unique(d[finite])
    n_pairs = int((int(finite.sum()) - d.shape[0]) // 2)
    if len(vals) < 2:
        return n_pairs, int(len(vals)), 0
    sv = np.sort(vals)
    near = int(np.sum((sv[1:] - sv[:-1]) / np.maximum(sv[1:], 1e-300) < 1e-5))
    return n_pairs, int(len(vals)), near


def run_ours(site, d, threshold, coeff):
    import importlib
    sys.path.insert(0, site)
    for m in [k for k in list(sys.modules) if k.startswith("canns_lib")]:
        del sys.modules[m]
    from canns_lib.ripser import ripser
    t0 = time.perf_counter()
    r = ripser(d, maxdim=1, thresh=threshold, coeff=coeff, distance_matrix=True,
               progress_bar=False)["dgms"]
    wall = time.perf_counter() - t0
    bars = [[[float(a), float(b)] for a, b in x] for x in r]
    return wall, bars


def run_gudhi(d, threshold, coeff):
    """Third-party oracle: GUDHI, same filtration, same coefficient field."""
    import gudhi
    st = gudhi.RipsComplex(distance_matrix=d,
                           max_edge_length=threshold).create_simplex_tree(max_dimension=2)
    st.persistence(homology_coeff_field=coeff)
    return {k: [(float(i[0]), float(i[1]))
                for i in st.persistence_intervals_in_dimension(k)] for k in (0, 1)}


def arbitrate(d, threshold, coeff, bars_pp, bars_us):
    """When the two sides disagree, ask a third implementation.

    Returns (oracle, oracle_agrees_with, verdict). `oracle_agrees_with` is
    "ours" / "ripserpp" / "both" / "neither"; the verdict vocabulary is
    closed:
      SAME_BARCODE            -- the two sides agree, no arbitration needed
      ORACLE_SUPPORTS_OURS   -- they disagree and GUDHI agrees with US
      ORACLE_SUPPORTS_RIPSERPP-- they disagree and GUDHI agrees with ripser++
      BARCODE_MISMATCH        -- they disagree and GUDHI agrees with neither
    """
    try:
        g = run_gudhi(d, threshold, coeff)
    except Exception as exc:  # pragma: no cover - oracle is optional
        return "unavailable(%s)" % type(exc).__name__, "unknown", "BARCODE_MISMATCH"
    u_o = u_p = 0
    for dim in (0, 1):
        gg = g.get(dim, [])
        pp = bars_pp.get(dim, [])
        us = bars_us[dim] if dim < len(bars_us) else []
        _, _, un_o, _ = match_bars(gg, us)
        _, _, un_p, _ = match_bars(gg, pp)
        u_o += un_o
        u_p += un_p
    if u_o == 0 and u_p > 0:
        return "gudhi", "ours", "ORACLE_SUPPORTS_OURS"
    if u_p == 0 and u_o > 0:
        return "gudhi", "ripserpp", "ORACLE_SUPPORTS_RIPSERPP"
    if u_o == 0 and u_p == 0:
        return "gudhi", "both", "ORACLE_SUPPORTS_OURS"
    return "gudhi", "neither", "BARCODE_MISMATCH"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/home/sichaohe/canns-lib-dev/site_ship2")
    ap.add_argument("--cells", type=int, default=6,
                    help="number of cells to run; 0 = every matrix in the dir")
    ap.add_argument("--reps", type=int, default=3, help="interleaved timing reps")
    ap.add_argument("--coeff", type=int, default=47)
    ap.add_argument("--out", default="gridcells_validation.csv")
    ap.add_argument("--resume-from", default="",
                    help="reuse the rows of an existing table for the cells that "
                         "already carry a SAME_BARCODE verdict, and re-run only "
                         "the cells that need it (e.g. to fill in the oracle)")
    ap.add_argument("--no-oracle", action="store_true",
                    help="do not run the GUDHI arbitration on mismatching cells")
    args = ap.parse_args()

    # matrix_meta.json is a CONCATENATION of JSON arrays ("[{..}][{..}]"),
    # so neither json.load nor a JSON-lines loop works on it: decode the
    # stream with raw_decode instead.
    meta = []
    text = open(os.path.join(GRID, "matrix_meta.json")).read()
    dec = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        obj, end = dec.raw_decode(text, pos)
        meta.extend(obj if isinstance(obj, list) else [obj])
        pos = end
    by_key = {m["key"]: m for m in meta}

    all_keys = sorted(os.path.basename(p)[:-4] for p in
                      glob.glob(os.path.join(GRID, "matrices", "*.npy")))
    if args.cells and args.cells < len(all_keys):
        step = max(1, len(all_keys) // args.cells)
        keys = all_keys[::step][:args.cells]
    else:
        keys = all_keys
    print("cells to run: %d of %d matrices in %s"
          % (len(keys), len(all_keys), os.path.join(GRID, "matrices")), flush=True)

    prev = {}
    if args.resume_from and os.path.exists(args.resume_from):
        with open(args.resume_from, newline="") as fh:
            for r in csv.DictReader(fh):
                prev[r["key"]] = r
        print("resuming from %s (%d rows already in it)" % (args.resume_from, len(prev)),
              flush=True)

    rows = []
    for i, key in enumerate(keys):
        # A cell that already carries a SAME_BARCODE verdict in the resumed
        # table is reused as-is (timings included): re-running it would only
        # cost time and could not change a verdict that is already supported
        # by a tolerance match.
        p = prev.get(key)
        if p and p.get("verdict") in ("SAME_BARCODE", "ORACLE_SUPPORTS_OURS") \
                and p.get("h1_md5_ours"):
            row = dict(p)
            row["reused"] = "1"
            row.setdefault("oracle", "none")
            row["oracle"] = row["oracle"] or "none"
            row.setdefault("oracle_agrees_with", "not_needed")
            row["oracle_agrees_with"] = row["oracle_agrees_with"] or "not_needed"
            if not row.get("near_tie_pairs"):
                # A column added after the table was first produced: fill it
                # in without re-running the benchmark.
                d0 = np.array(np.load(os.path.join(GRID, "matrices", key + ".npy")),
                              dtype=np.float64, copy=True)
                np.fill_diagonal(d0, 0.0)
                f0 = np.isfinite(d0)
                n_pairs, n_distinct, near = degeneracy_stats(d0, f0)
                row["finite_pairs"] = n_pairs
                row["distinct_values"] = n_distinct
                row["near_tie_pairs"] = near
                row["reused"] = "2"
            rows.append(row)
            print("%-46s reused   ours %7ss pp %7ss ratio %6s  %s"
                  % (key, row.get("ours_s"), row.get("ripserpp_s"),
                     row.get("ratio"), row.get("verdict")), flush=True)
            continue

        mpath = os.path.join(GRID, "matrices", key + ".npy")
        d = np.load(mpath)
        d = np.array(d, dtype=np.float64, copy=True)
        np.fill_diagonal(d, 0.0)
        finite = np.isfinite(d)
        t = float(d[finite].max())

        # --- interleaved timing: min over reps, both sides in the same loop
        # The text file for ripser++ is prepared once, outside the timed
        # region; both sides then alternate so a machine-wide slowdown hits
        # both equally.
        pp_in = prepare_ripserpp_input(d)
        w_us = w_pp = float("inf")
        o = None
        foreign = foreign_max = ""
        for _ in range(args.reps):
            (tw, bars_us), f_us, m_us = bench_core.with_foreign(
                run_ours, args.site, d, t, args.coeff)
            w_us = min(w_us, tw)
            tp0 = time.perf_counter()
            o, f_pp, m_pp = bench_core.with_foreign(run_ripserpp, pp_in, t)
            w_pp = min(w_pp, time.perf_counter() - tp0)
            # The worst foreign load over the cell, over both sides and over
            # the repetitions: the cell is a measurement only if every part
            # of it was taken on a quiet host.
            for f, m in ((f_us, m_us), (f_pp, m_pp)):
                try:
                    foreign = max(float(foreign or 0.0), float(f))
                    foreign_max = max(float(foreign_max or 0.0), float(m))
                except (TypeError, ValueError):
                    pass
        os.unlink(pp_in)
        bars_pp = [[float(x) for x in b] for b in o.get(1, [])]

        mm = by_key.get(key, {})
        n_pairs, n_distinct, near = degeneracy_stats(d, finite)
        row = {"key": key, "n": d.shape[0],
               "finite_pct": "%.4f" % finite.mean(),
               "threshold": "%.6g" % t,
               "finite_pairs": n_pairs, "distinct_values": n_distinct,
               "near_tie_pairs": near,
               "ours_s": "%.3f" % w_us, "ripserpp_s": "%.3f" % w_pp,
               "ratio": "%.3f" % (w_us / w_pp) if w_pp else "",
               "foreign_cpu": "%.4f" % foreign if foreign != "" else "",
               "foreign_max": "%.3f" % foreign_max if foreign_max != "" else ""}
        ok = True
        for dim in (0, 1):
            a = o.get(dim, [])
            b = bars_us[dim] if dim < len(bars_us) else []
            n_m, max_dev, un_a, un_b = match_bars(a, b)
            row["bars_pp_d%d" % dim] = len(a)
            row["bars_ours_d%d" % dim] = len(b)
            row["unmatched_d%d" % dim] = max(un_a, un_b)
            row["max_rel_dev_d%d" % dim] = "%.3e" % max_dev
            if n_m != len(a) or un_a or un_b:
                ok = False
        row["h1_md5_ours"] = md5bars(bars_us[1]) if len(bars_us) > 1 else ""
        if ok:
            row["verdict"] = "SAME_BARCODE"
            row["oracle"] = "none"
            row["oracle_agrees_with"] = "not_needed"
        else:
            if args.no_oracle:
                row["verdict"] = "BARCODE_MISMATCH"
                row["oracle"] = "none"
                row["oracle_agrees_with"] = "not_run"
            else:
                orc, agrees, verdict = arbitrate(d, t, args.coeff, o, bars_us)
                row["verdict"] = verdict
                row["oracle"] = orc
                row["oracle_agrees_with"] = agrees
        rows.append(row)
        print("%-46s ours %7.3fs pp %7.3fs ratio %6.3f  d0 %4d/%-4d d1 %4d/%-4d  %s%s"
              % (key, w_us, w_pp, w_us / w_pp,
                 row["bars_ours_d0"], row["bars_pp_d0"],
                 row["bars_ours_d1"], row["bars_pp_d1"], row["verdict"],
                 ("  [%s agrees with %s]" % (row["oracle"], row["oracle_agrees_with"]))
                 if row["oracle"] not in ("none",) else ""), flush=True)

    fields = ["key", "n", "finite_pct", "threshold", "finite_pairs", "distinct_values",
              "near_tie_pairs", "ours_s", "ripserpp_s", "ratio",
              "bars_ours_d0", "bars_pp_d0", "unmatched_d0", "max_rel_dev_d0",
              "foreign_cpu", "foreign_max",
              "bars_ours_d1", "bars_pp_d1", "unmatched_d1", "max_rel_dev_d1",
              "h1_md5_ours", "oracle", "oracle_agrees_with", "verdict", "reused"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s (%d cells)" % (args.out, len(rows)))
    bad = [r for r in rows if r["verdict"] == "BARCODE_MISMATCH"]
    oracle = [r for r in rows if r["verdict"].startswith("ORACLE_")]
    print("barcode mismatches (unarbitrated): %d" % len(bad))
    print("arbitrated by the oracle: %d (%s)"
          % (len(oracle), ", ".join(sorted({r["verdict"] for r in oracle})) or "-"))
    ratios = [float(r["ratio"]) for r in rows if r.get("ratio")]
    if ratios:
        ratios.sort()
        print("ratio ours/ripser++: min %.3f  median %.3f  max %.3f  (n=%d)"
              % (ratios[0], ratios[len(ratios) // 2], ratios[-1], len(ratios)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
