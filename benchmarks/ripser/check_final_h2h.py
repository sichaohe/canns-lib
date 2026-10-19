#!/usr/bin/env python
"""Checker for the final head-to-head CSV.

The CSV is the single source of truth; the markdown deliverable is a view of
it. This checker is the thing that says PASS or FAIL, and it computes every
denominator itself from the EXPECTED grid -- it never takes a count on faith
from the runner or from prose.

WHAT IS CHECKED
  1. COMPLETENESS. Every (arm, rep) cell of the EXPECTED grid is present
     exactly once. A missing or duplicated cell is a FAIL, not a footnote.
  2. NO SILENT FALLBACK. A ripser++ run that fell back to CPU-only is
     rejected outright; its number must not enter a median.
  3. BARCODE AGREEMENT. All canns arms must produce the same per-dimension
     md5, and that md5 must equal the recorded reference. The canns arms
     differ only in WHERE the sort happens, so any disagreement is a bug.
  4. CROSS-IMPLEMENTATION AGREEMENT. ripser++ must report the same bar
     COUNTS and the same rounded H1 top-3 lifetimes. (Its md5 cannot be
     compared: it prints 7 significant digits, we print f32 values.)
  5. NO IMPLAUSIBLE TIMING. A run whose wall is outside [0.2x, 5x] the arm
     median is a FAIL, not an outlier to be silently dropped.

The `--self-test` mode tampers with a copy of the CSV and asserts the
checker REJECTS each tamper. A checker that only ever prints PASS proves
nothing about itself.

USAGE
    python check_final_h2h.py <csv> [--self-test]
"""

import argparse
import csv
import os
import statistics
import sys
import tempfile

ARMS = [
    "ours_gpu_radix",
    "ours_gpu_hostsort",
    "ripserpp_build2",
    "ripserpp_build3",
    "ours_gpu_radix_inf",
]
REFERENCE_MD5 = (
    "ee1f5283091dd645a3dd53e6f1a7eac6:"
    "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
    "2d018d8fe9c1895b1b74f837af59ed20"
)
# The inf-preserved arm has a different barcode: excluding the +inf pairs
# removes simplices that the capped run legitimately contains. It is
# compared for TIMING only.
#
# The ripser++ arms are exempt for a different reason: ripser++ prints 7
# significant digits and we print f32 values, so an md5 comparison between
# the two is meaningless -- it would fail on a rounding difference, not on a
# real disagreement. Their agreement is checked through the bar COUNTS and
# the rounded H1 top-3 lifetimes instead (see the cross-implementation
# check below).
MD5_EXEMPT = {"ours_gpu_radix_inf", "ripserpp_build2", "ripserpp_build3"}


def expected_grid(reps):
    return [(arm, rep) for arm in ARMS for rep in range(1, reps + 1)]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check(rows, reps, verbose=True):
    """Returns a list of failure strings (empty means PASS)."""
    fails = []
    if not rows:
        return ["the CSV is empty"]

    exp = expected_grid(reps)
    seen = {}
    for r in rows:
        key = (r["arm"], int(r["rep"]))
        if key in seen:
            fails.append(f"duplicate cell {key}")
        seen[key] = r
    for key in exp:
        if key not in seen:
            fails.append(f"missing cell {key}")
    extra = set(seen) - set(exp)
    for key in sorted(extra):
        fails.append(f"unexpected cell {key} (not in the EXPECTED grid)")

    # --- fallback -------------------------------------------------------
    for r in rows:
        fb = (r.get("fallback") or "").strip().lower()
        if fb in ("true", "1"):
            fails.append(
                f"{r['arm']} rep{r['rep']} fell back to CPU-only; the timing "
                "must not be used"
            )
    # --- walls ----------------------------------------------------------
    for r in rows:
        try:
            w = float(r["wall"])
        except (KeyError, ValueError):
            fails.append(f"{r['arm']} rep{r['rep']} has no usable wall")
            continue
        if not (0.05 < w < 3600):
            fails.append(f"{r['arm']} rep{r['rep']} wall {w} is implausible")

    medians = {}
    for arm in ARMS:
        ws = []
        for r in rows:
            if r["arm"] != arm:
                continue
            try:
                ws.append(float(r["wall"]))
            except (KeyError, ValueError):
                pass
        if ws:
            medians[arm] = statistics.median(ws)

    # --- barcode agreement ----------------------------------------------
    by_arm_md5 = {}
    for r in rows:
        if r["arm"] in MD5_EXEMPT:
            continue
        by_arm_md5.setdefault(r["arm"], set()).add(r.get("md5", ""))
    for arm, s in by_arm_md5.items():
        if len(s) != 1:
            fails.append(f"{arm} produced {len(s)} different barcodes across reps: {s}")
        elif s and REFERENCE_MD5 not in s:
            fails.append(
                f"{arm} barcode {list(s)[0]} != reference {REFERENCE_MD5}"
            )

    # --- cross-implementation agreement ----------------------------------
    ours = [r for r in rows if r["arm"] == "ours_gpu_radix"]
    rps = [r for r in rows if r["arm"].startswith("ripserpp")]
    if ours and rps:
        ob = {r.get("bars") for r in ours}
        rb = {r.get("bars") for r in rps}
        if len(ob) != 1 or len(rb) != 1:
            fails.append(f"bar counts not stable: ours={ob} ripserpp={rb}")
        elif ob != rb:
            fails.append(f"bar counts differ: ours={ob} ripserpp={rb}")
        oh = {r.get("h1top") for r in ours}
        rh = {r.get("h1top") for r in rps}
        # Numeric comparison with a tolerance: ripser++ prints 7 significant
        # digits and we print f32, so a value sitting on a rounding boundary
        # can differ in the second decimal by construction. An exact string
        # comparison would flag that as a disagreement that is not there.
        def nums(s):
            out = []
            for v in (s or "").split(","):
                try:
                    out.append(float(v))
                except ValueError:
                    out.append(None)
            return out
        on, rn = nums(list(oh)[0] if oh else ""), nums(list(rh)[0] if rh else "")
        if len(on) != len(rn) or any(
            a is None or b is None or abs(a - b) > 0.011 for a, b in zip(on, rn)
        ):
            fails.append(
                f"H1 top-3 lifetimes differ beyond 0.01: ours={oh} ripserpp={rh}"
            )

    # --- plausibility of each arm's own spread ---------------------------
    for arm in ARMS:
        ws = [float(r["wall"]) for r in rows
              if r["arm"] == arm and r.get("wall")]
        if len(ws) >= 2:
            med = statistics.median(ws)
            for r in rows:
                if r["arm"] != arm:
                    continue
                w = float(r["wall"])
                if not (0.2 * med <= w <= 5.0 * med):
                    fails.append(
                        f"{arm} rep{r['rep']} wall {w:.3f} is outside "
                        f"[0.2x, 5x] the arm median {med:.3f}"
                    )

    if verbose:
        print("EXPECTED_CELLS %d (arms=%d x reps=%d)" % (len(exp), len(ARMS), reps))
        print("ROWS %d" % len(rows))
        for arm in ARMS:
            if arm in medians:
                ws = [float(r["wall"]) for r in rows if r["arm"] == arm]
                print("  %-20s median=%8.3f min=%8.3f max=%8.3f n=%d"
                      % (arm, statistics.median(ws), min(ws), max(ws), len(ws)))
        if "ripserpp_build2" in medians and "ours_gpu_radix" in medians:
            print("  ours_gpu_radix / ripserpp(build2) = %.3fx (capped, comparable)"
                  % (medians["ours_gpu_radix"] / medians["ripserpp_build2"]))
        if "ripserpp_build3" in medians and "ours_gpu_radix" in medians:
            print("  ours_gpu_radix / ripserpp(build3) = %.3fx (capped, comparable)"
                  % (medians["ours_gpu_radix"] / medians["ripserpp_build3"]))
        if "ours_gpu_hostsort" in medians and "ours_gpu_radix" in medians:
            print("  hostsort / radix        = %.3fx (speedup from the device sort)"
                  % (medians["ours_gpu_hostsort"] / medians["ours_gpu_radix"]))
    return fails


def self_test(path, reps):
    """Tamper with a copy of the CSV and assert the checker rejects it."""
    base = load(path)
    tampers = []

    def t(name, fn):
        tampers.append((name, fn))

    def drop_row(rows):
        r = [x for x in rows]
        return r[:-1]

    def dup_row(rows):
        r = list(rows)
        r.append(dict(rows[0]))
        return r

    def break_md5(rows):
        r = [dict(x) for x in rows]
        r[0]["md5"] = "deadbeef" * 4
        return r

    def set_fallback(rows):
        r = [dict(x) for x in rows]
        for x in r:
            if x["arm"] == "ripserpp_build2":
                x["fallback"] = "True"
                break
        return r

    def break_bars(rows):
        r = [dict(x) for x in rows]
        for x in r:
            if x["arm"] == "ripserpp_build2":
                x["bars"] = "1|2|3"
                break
        return r

    def absurd_wall(rows):
        r = [dict(x) for x in rows]
        r[0]["wall"] = "0.001"
        return r

    t("a row is missing", drop_row)
    t("a cell is duplicated", dup_row)
    t("a barcode md5 is wrong", break_md5)
    t("a ripser++ run silently fell back to CPU", set_fallback)
    t("the bar counts disagree with ripser++", break_bars)
    t("a wall time is implausible", absurd_wall)

    ok = True
    for name, fn in tampers:
        rows = fn(base)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(base[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
            tmp = f.name
        try:
            fails = check(load(tmp), reps, verbose=False)
        finally:
            os.unlink(tmp)
        if not fails:
            print("  SELF-TEST FAILED: the checker accepted %r" % name)
            ok = False
        else:
            print("  rejected as it must be: %-44s -> %s" % (name, fails[0]))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        print("=== negative self-test ===")
        good = self_test(args.csv, args.reps)
        print("SELF-TEST", "PASS" if good else "FAIL")
        return 0 if good else 1

    fails = check(load(args.csv), args.reps)
    if fails:
        print("FAIL")
        for f in fails:
            print("  ! " + f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
