#!/usr/bin/env python3
"""Is the T arm really the same filtration? -- and are the barcodes equal?

THE QUESTION. ripser++ cannot read a matrix with a missing distance, so
it can only be run on the CAPPED matrix. But it can be run with the
threshold at the maximum finite distance, and then the capped pairs are
excluded -- which is the SAME filtration as the one we get by keeping the
`+inf`. That equivalence is the whole basis of the fair comparison, and
it has to be MEASURED, not asserted: a reviewer will ask what happens to
the bars that die above the threshold.

WHAT THIS SCRIPT MEASURES, per dimension, with the bar VALUES and not
only the bar COUNTS:

  dying        ours: the bars whose death is at or below T. These must
               match ripser++'s finite bars one for one, in birth and in
               death, within the tolerance the reference's six printed
               digits allow.
  truncated    ours: the bars whose death is above T but finite. In the T
               arm ripser++ cannot see those deaths, so it reports them
               as essential.
  essential    ours: the bars whose death is genuinely infinite. Also
               reported as essential by the T arm.

  And the T arm's own essential bars -- the ones ripser++ prints with an
  infinite death -- must be exactly our (truncated + essential) bars,
  matched on their BIRTHS (their deaths are infinite on both sides by
  construction, so only the births carry information).

So the claim being tested is precise: **the T arm and the `+inf`
representation compute the same filtration, and their barcodes agree
value by value, with the only difference being that the T arm cannot
name the deaths above the threshold.**

USAGE
    python t_arm_equivalence.py --out t_arm_equivalence.csv [--sizes 1200,1600]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_scan import PY, PP, make_instance, run_timed  # noqa: E402

RTOL = 1e-5
# ripser++ prints 6 significant digits -> a relative error up to 5e-7 (up
# to 5e-6 for values < 1). 1e-5 leaves a 2x margin and still rejects a
# genuinely different filtration.

FIELDS = ["n", "dim", "T", "ours_dying", "ours_truncated", "ours_essential",
          "pp_finite", "pp_essential", "matched", "unmatched_ours",
          "unmatched_pp", "max_rel_dev", "birth_matched", "birth_unmatched",
          "verdict", "loadavg"]

CHILD = r'''
import json, sys, time
import numpy as np
site, path, maxdim, coeff = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                             int(sys.argv[4]))
sys.path.insert(0, site)
from canns_lib.ripser import ripser
d = np.load(path)
t0 = time.perf_counter()
r = ripser(d, maxdim=maxdim, thresh=float("inf"), coeff=coeff,
           distance_matrix=True, progress_bar=False)["dgms"]
wall = time.perf_counter() - t0
print("WALL %.4f" % wall)
print("BARS " + json.dumps({str(d): [list(map(float, b)) for b in x]
                             for d, x in enumerate(r)}))
'''


def parse_reference_intervals(path):
    """ripser++'s intervals, per dimension.

    The format is half-open "[birth,death)" and an essential bar is
    printed with an infinite death.
    """
    out, dim = {}, None
    for line in open(path):
        line = line.strip()
        m = re.match(r"persistence intervals in dim (\d+):", line)
        if m:
            dim = int(m.group(1))
            out.setdefault(dim, [])
            continue
        if dim is None or not line.startswith("["):
            continue
        body = line[1:].rstrip(")").rstrip("]")
        try:
            b, _, d = body.partition(",")
        except ValueError:
            continue
        try:
            birth = float(b)
        except ValueError:
            continue
        ds = d.strip()
        death = math.inf if ds in ("inf", "Infinity", "") else float(ds)
        out[dim].append((birth, death))
    return out


def match(a, b, rtol=RTOL):
    """Greedy nearest-neighbour match of two bar lists.

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
        d = np.abs(x - y)
        d[both_inf] = 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(np.isfinite(y) & (y != 0), d / np.abs(y), d)
        dev = np.maximum(dev, rel)
    order = np.argsort(dev, axis=None)
    used_a, used_b, matched, maxdev = set(), set(), 0, 0.0
    for flat in order:
        i, j = divmod(int(flat), len(B))
        if i in used_a or j in used_b:
            continue
        if dev[i, j] > rtol:
            continue
        used_a.add(i)
        used_b.add(j)
        matched += 1
        maxdev = max(maxdev, float(dev[i, j]))
    return matched, maxdev, len(a) - matched, len(b) - matched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/home/sichaohe/canns-lib-dev/site_ship6")
    ap.add_argument("--sizes", default="1200,1600")
    ap.add_argument("--dim", type=int, default=9)
    ap.add_argument("--missing", type=float, default=0.30)
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--coeff", type=int, default=47)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="t_arm_equivalence.csv")
    args = ap.parse_args()

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CANNS_BENCH_GPU", "0"))
    rows = []
    for n in [int(x) for x in args.sizes.split(",") if x.strip()]:
        p_inf = "/tmp/_teq_%d_inf.npy" % n
        p_cap = "/tmp/_teq_%d_cap.npy" % n
        p_txt = "/tmp/_teq_%d.txt" % n
        T, inf_frac = make_instance(n, args.dim, args.missing, 1234 + n,
                                    p_inf, p_cap, p_txt)
        print("=== n=%d  missing pairs %.4f  T=%.6g ===" % (n, inf_frac, T),
              flush=True)

        # --- ours, on the matrix with the +inf kept, threshold = inf ------
        out_text, _e, rc = run_timed(
            ["/usr/bin/time", "-f", "__T__%e %M", PY, "-c", CHILD, args.site,
             p_inf, str(args.maxdim), str(args.coeff)],
            "/tmp/_teq_ours.out", args.timeout, env=env)
        ours = {}
        for line in out_text.splitlines():
            if line.startswith("BARS "):
                import json
                ours = {int(k): [tuple(map(float, b)) for b in v]
                        for k, v in json.loads(line[5:]).items()}
        if not ours:
            print("FAIL: no barcode from our side at n=%d" % n, file=sys.stderr)
            return 1

        # --- ripser++, on the capped matrix, threshold = T -----------------
        run_timed(["/usr/bin/time", "-f", "__T__%e %M", PP,
                   "--format", "distance", "--dim", str(args.maxdim),
                   "--threshold", "%.10g" % T, "--ratio", "1.0", p_txt],
                  "/tmp/_teq_pp.out", args.timeout, env=env)
        pp = parse_reference_intervals("/tmp/_teq_pp.out")

        for d in range(args.maxdim + 1):
            ob = ours.get(d, [])
            pb = pp.get(d, [])
            dying = [x for x in ob if x[1] <= T]
            trunc = [x for x in ob if T < x[1] < math.inf]
            ess = [x for x in ob if math.isinf(x[1])]
            pp_fin = [x for x in pb if not math.isinf(x[1])]
            pp_ess = [x for x in pb if math.isinf(x[1])]

            m, dev, ua, ub = match(dying, pp_fin)
            # the essential bars: the deaths are infinite on both sides, so
            # only the births carry information
            bm, _bdev, bua, bub = match([(x[0], 0.0) for x in trunc + ess],
                                         [(x[0], 0.0) for x in pp_ess])
            ok = (ua == 0 and ub == 0 and bua == 0 and bub == 0
                  and len(trunc) + len(ess) == len(pp_ess))
            rows.append({
                "n": n, "dim": d, "T": "%.6g" % T,
                "ours_dying": len(dying), "ours_truncated": len(trunc),
                "ours_essential": len(ess),
                "pp_finite": len(pp_fin), "pp_essential": len(pp_ess),
                "matched": m, "unmatched_ours": ua, "unmatched_pp": ub,
                "max_rel_dev": "%.3e" % dev,
                "birth_matched": bm, "birth_unmatched": bua + bub,
                "verdict": "SAME_FILTRATION" if ok else "DIFFERENT",
                "loadavg": open("/proc/loadavg").read().split()[0],
            })
            print("  dim %d: ours dying=%d truncated=%d essential=%d | "
                  "pp finite=%d essential=%d | matched=%d dev=%s | %s"
                  % (d, len(dying), len(trunc), len(ess), len(pp_fin),
                     len(pp_ess), m, rows[-1]["max_rel_dev"],
                     rows[-1]["verdict"]), flush=True)
        for f in (p_inf, p_cap, p_txt):
            if os.path.exists(f):
                os.unlink(f)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    bad = [r for r in rows if r["verdict"] != "SAME_FILTRATION"]
    print("\nwrote %s (%d rows); different filtrations: %d"
          % (args.out, len(rows), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
