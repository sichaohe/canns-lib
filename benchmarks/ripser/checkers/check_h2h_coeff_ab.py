#!/usr/bin/env python3
"""Checker for the coefficient A/B table (h2h_coeff_ab.csv).

The table holds, per (repetition, arm), one run of the ASA head-to-head:
our build over Z/47, our build over Z/2, and ripser++ over Z/2, with the
arms interleaved and the order rotating on every repetition.

The claims it has to support:

1. **the grid is complete** -- three arms, the same number of repetitions
   each, and every repetition has all three;
2. **the host was quiet** -- the foreign load on every row is below the
   threshold;
3. **every run read its input** -- the bar count is the expected one. This
   is the check that catches a reference run that read nothing: it comes
   back in a fraction of the time and with a handful of bars, and a table
   that reports it reports a failure as a result;
4. **the coefficient does not change the measurement** -- our two arms
   agree on the barcodes exactly and their times agree within the
   tolerance, which is what makes the Z/2 comparison legitimate;
5. **the run is interleaved** -- the arms appear in a different order in
   different repetitions, so a drift in the machine cannot favour one of
   them.

It prints its own denominator and carries negative self-tests.

USAGE
    python check_h2h_coeff_ab.py h2h_coeff_ab.csv [--max-foreign-cpu 0.25]
    python check_h2h_coeff_ab.py h2h_coeff_ab.csv --self-test
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

ARMS = ("ours_z47", "ours_z2", "ripserpp")

# The ASA matrix's homology in dimensions 0 / 1 / 2, as our build reports
# it (and as ripser++ reports it in its one-per-line format, 1734 in
# total). These are the numbers the comparison is only meaningful at.
OURS_BARS = "1195|481|58"
PP_BARS_TOTAL = 1734

# How far apart our two arms may be before the coefficient is doing
# something to the measurement.
COEFF_TOLERANCE = 0.05

REQUIRED = ("rep", "arm", "wall_s", "bars", "foreign_cpu", "verdict")


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    by = {}
    for r in rows:
        by.setdefault(r["rep"], {})[r["arm"]] = r
    return rows, by


def check(path, max_foreign_cpu=0.25):
    fails = []
    rows, by = load(path)
    if not rows:
        return ["the table is empty: %s" % path]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return ["the table is missing columns: %s" % ", ".join(missing_cols)]

    reps = sorted(by)
    expected = len(reps) * len(ARMS)
    for rep in reps:
        for arm in ARMS:
            if arm not in by[rep]:
                fails.append("missing cell: rep %s / %s" % (rep, arm))

    # the quiet host
    for r in rows:
        if (r.get("verdict") or "").strip() != "OK":
            fails.append("rep %s / %s: the run did not succeed (verdict %r)"
                         % (r["rep"], r["arm"], r.get("verdict")))
        raw = (r.get("foreign_cpu") or "").strip()
        if not raw:
            fails.append("rep %s / %s: no foreign-load reading, so the table "
                         "cannot show it was measured on a quiet host"
                         % (r["rep"], r["arm"]))
            continue
        try:
            value = float(raw)
        except ValueError:
            fails.append("rep %s / %s: the foreign load is not a number (%r)"
                         % (r["rep"], r["arm"], raw))
            continue
        if max_foreign_cpu > 0 and value > max_foreign_cpu:
            fails.append("rep %s / %s: the host was used by other users at "
                         "%.3f cores, above the %.2f-core threshold"
                         % (r["rep"], r["arm"], value, max_foreign_cpu))

    # every run read its input
    for r in rows:
        arm = r["arm"]
        bars = (r.get("bars") or "").strip()
        if not bars:
            fails.append("rep %s / %s: no bar count -- the run read nothing"
                         % (r["rep"], arm))
            continue
        if arm in ("ours_z47", "ours_z2"):
            if bars != OURS_BARS:
                fails.append("rep %s / %s: the bar count is %r, not %r"
                             % (r["rep"], arm, bars, OURS_BARS))
        else:
            try:
                total = sum(int(x) for x in bars.split("|"))
            except ValueError:
                fails.append("rep %s / %s: the bar count %r is not a list of "
                             "numbers" % (r["rep"], arm, bars))
                continue
            if total != PP_BARS_TOTAL:
                fails.append("rep %s / %s: the reference reports %d bars, not "
                             "%d -- it did not read the same instance"
                             % (r["rep"], arm, total, PP_BARS_TOTAL))

    # the coefficient does not change the measurement
    for rep in reps:
        a, b = by[rep].get("ours_z47"), by[rep].get("ours_z2")
        if not a or not b:
            continue
        if (a.get("bars") or "").strip() != (b.get("bars") or "").strip():
            fails.append("rep %s: the two coefficient fields give different "
                         "barcodes (%r and %r) -- they are not the same "
                         "computation" % (rep, a.get("bars"), b.get("bars")))
        if a.get("md5") and b.get("md5") and a["md5"] != b["md5"]:
            fails.append("rep %s: the two coefficient fields give different "
                         "barcode md5s" % rep)

    def med(arm):
        vals = []
        for rep in reps:
            r = by[rep].get(arm)
            if r:
                try:
                    vals.append(float((r.get("wall_s") or "").strip()))
                except ValueError:
                    pass
        return vals

    z47, z2 = med("ours_z47"), med("ours_z2")
    if z47 and z2:
        ratio = sorted(z47)[len(z47) // 2] / sorted(z2)[len(z2) // 2]
        if abs(ratio - 1.0) > COEFF_TOLERANCE:
            fails.append("our Z/47 and our Z/2 differ by %.1f%% -- the "
                         "coefficient changes what is measured, so the two "
                         "are not the same computation"
                         % (100.0 * (ratio - 1.0)))

    # the arms were interleaved
    orders = set()
    for rep in reps:
        order = [r["arm"] for r in rows if r["rep"] == rep]
        orders.add(tuple(order))
    if len(reps) > 1 and len(orders) == 1:
        fails.append("every repetition ran the arms in the same order, so a "
                     "drift in the machine would favour the same arm every "
                     "time -- the arms have to be interleaved")

    return fails


MUTATIONS = ["drop_row", "busy_host", "no_foreign", "fail_verdict",
             "wrong_bars", "reference_read_nothing", "not_interleaved"]


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    if kind == "drop_row":
        rows = [r for r in rows if not (r["arm"] == "ours_z2" and r["rep"] == "3")]
    elif kind == "busy_host":
        for r in rows:
            r["foreign_cpu"] = "1.500"
    elif kind == "no_foreign":
        for r in rows:
            r["foreign_cpu"] = ""
    elif kind == "fail_verdict":
        for r in rows:
            if r["arm"] == "ours_z47" and r["rep"] == "2":
                r["verdict"] = "RC=1"
    elif kind == "wrong_bars":
        for r in rows:
            if r["arm"] == "ours_z2":
                r["bars"] = "1195|481|57"
    elif kind == "reference_read_nothing":
        for r in rows:
            if r["arm"] == "ripserpp":
                r["bars"] = "7|109|153"
    elif kind == "not_interleaved":
        rows.sort(key=lambda r: (r["arm"], r["rep"]))
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="h2h_coeff_ab.csv")
    ap.add_argument("--max-foreign-cpu", type=float, default=0.25)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    rows, by = load(args.csv)
    fails = check(args.csv, max_foreign_cpu=args.max_foreign_cpu)
    reps = len(by)
    print("expected cells : %d repetitions x %d arms = %d"
          % (reps, len(ARMS), reps * len(ARMS)))
    print("rows in table  : %d" % len(rows))
    for rep in sorted(by):
        print("  rep %s: %s" % (rep, ",".join(sorted(by[rep]))))
    if fails:
        print("\nFAIL (%d problem(s)):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: the grid is complete, every run read its input (the "
          "reference reports the expected 1734 bars), the host was quiet, the "
          "two coefficient fields give the same barcodes and the same time, "
          "and the arms were interleaved")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="coeffab_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, max_foreign_cpu=args.max_foreign_cpu)
                if f:
                    print("  self-test %-22s -> correctly FAILED (%d problem(s))"
                          % (kind, len(f)))
                else:
                    print("  self-test %-22s -> *** NOT DETECTED ***" % kind)
                    bad += 1
            if bad:
                print("\nFAIL: %d mutation(s) slipped through" % bad)
                return 1
            print("\nPASS: all %d negative self-tests detected" % len(MUTATIONS))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
