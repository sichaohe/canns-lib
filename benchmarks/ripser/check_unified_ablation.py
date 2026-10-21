#!/usr/bin/env python3
"""Checker for the unified ablation table.

The table is "all switches on, against exactly one off", so:

  1. THE GRID IS COMPLETE. Every variant x every arm x every repetition,
     and the baseline ("all_on") has to be there for every arm -- without
     it there is nothing to compare against. The denominator is computed
     from the table's own contents.
  2. EVERY RUN REPRODUCED THE REFERENCE BARCODE. An ablation row that
     changed the answer is not an ablation row, it is a bug.
  3. THE MEASUREMENTS ARE IDENTIFIED. The load average, the device, the
     source fingerprint and the build are on every row.
  4. THE EFFECT IS STATED WITH ITS NOISE. The checker prints, for each
     variant and arm, the median of the variant over the median of the
     baseline, and compares the effect with the spread of the baseline's
     own measurements: an effect smaller than the measurement's own spread
     is reported as "within noise", not as a win.

USAGE
    python check_unified_ablation.py unified_ablation.csv [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import sys
import tempfile

REQUIRED = ["rep", "variant", "feature", "arm", "build", "wall_s", "rss_kb",
            "bars", "md5", "verdict", "loadavg", "foreign_cpu", "foreign_max",
            "gpu", "commit"]
BASELINE = "all_on"


def check(path, min_reps=3, max_loadavg=0.0, max_foreign_cpu=0.25):
    fails = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["table is empty: %s" % path]
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        return ["table is missing columns: %s" % ", ".join(missing)]

    variants = sorted({r["variant"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    reps = sorted({int(r["rep"]) for r in rows})
    print("variants %s x arms %d x repetitions %d = %d rows expected"
          % (variants, len(arms), len(reps),
             len(variants) * len(arms) * len(reps)))
    print("rows in table                    : %d" % len(rows))

    by = {}
    for r in rows:
        key = (r["variant"], r["arm"], int(r["rep"]))
        if key in by:
            fails.append("duplicated row: %s / %s / rep %s" % key)
            continue
        by[key] = r

    for v in variants:
        for a in arms:
            got = [r for r in by.values() if r["variant"] == v and r["arm"] == a]
            if len(got) < min_reps:
                fails.append("%s / %s: %d repetition(s), the paper needs at "
                             "least %d" % (v, a, len(got), min_reps))
    for a in arms:
        if not [r for r in by.values()
                if r["variant"] == BASELINE and r["arm"] == a]:
            fails.append("no baseline: %s / %s is missing, and without it "
                         "there is nothing to compare against" % (BASELINE, a))

    for key, r in by.items():
        if r["verdict"] != "OK":
            fails.append("%s / %s / rep %s: verdict %s"
                         % (r["variant"], r["arm"], r["rep"], r["verdict"]))
        for col in ("loadavg", "foreign_cpu", "gpu", "commit", "build"):
            if not (r.get(col) or "").strip():
                fails.append("%s / %s / rep %s: the row does not record %s"
                             % (r["variant"], r["arm"], r["rep"], col))
        try:
            if float(r["wall_s"]) <= 0:
                fails.append("%s / %s / rep %s: non-positive wall time"
                             % (r["variant"], r["arm"], r["rep"]))
            if int(r["rss_kb"]) <= 0:
                fails.append("%s / %s / rep %s: no peak RSS"
                             % (r["variant"], r["arm"], r["rep"]))
        except ValueError:
            fails.append("%s / %s / rep %s: the timing columns are not numbers"
                         % (r["variant"], r["arm"], r["rep"]))

    # --- the effect of each variant, with its noise ------------------------
    print("\n| variant | switch | arm | baseline (s) | variant (s) | time | "
          "peak RSS | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for v in variants:
        if v == BASELINE:
            continue
        for a in arms:
            base = [float(by[(BASELINE, a, rep)]["wall_s"])
                    for rep in reps if (BASELINE, a, rep) in by]
            var = [float(by[(v, a, rep)]["wall_s"])
                   for rep in reps if (v, a, rep) in by]
            if not base or not var:
                continue
            bm, vm = statistics.median(base), statistics.median(var)
            spread = (max(base) - min(base)) / bm if bm else 0.0
            ratio = vm / bm if bm else 0.0
            rb = [int(by[(BASELINE, a, rep)]["rss_kb"]) / 1048576.0
                  for rep in reps if (BASELINE, a, rep) in by]
            rv = [int(by[(v, a, rep)]["rss_kb"]) / 1048576.0
                  for rep in reps if (v, a, rep) in by]
            rbm, rvm = statistics.median(rb), statistics.median(rv)
            feature = by[(v, a, reps[0])]["feature"] if (v, a, reps[0]) in by else ""
            verdict = ("within noise" if abs(ratio - 1.0) <= spread
                       else ("the switch is worth it" if ratio > 1.0
                             else "the switch costs time"))
            print("| %s | %s | %s | %.2f | %.2f | %.3fx | %.1f -> %.1f GB | %s |"
                  % (v, feature, a, bm, vm, ratio, rbm, rvm, verdict))

    # A table measured while the host was busy is not a measurement of the
    # code: on this shared machine a neighbour's job changes the numbers by
    # up to 40% and can flip a comparison's sign. So a table can be asked
    # to prove it was measured in a quiet window.
    if max_foreign_cpu > 0:
        worst_mean, worst_max = 0.0, 0.0
        bad = None
        for r in rows:
            raw = (r.get("foreign_cpu") or "").strip()
            if not raw:
                bad = "a row has no foreign-load reading"
                break
            try:
                fc = float(raw)
            except ValueError:
                bad = "the foreign-load column is not a number"
                break
            try:
                fm = float((r.get("foreign_max") or "").strip() or 0.0)
            except ValueError:
                fm = 0.0
            worst_mean = max(worst_mean, fc)
            worst_max = max(worst_max, fm)
            if fc > max_foreign_cpu and bad is None:
                bad = ("the foreign load %s cores on this row is above the "
                       "quiet-window threshold %.2f -- somebody else was "
                       "using the host while this row was measured"
                       % (raw, max_foreign_cpu))
        print("worst foreign load: mean %.3f cores, largest single window "
              "%.3f cores (threshold %.2f)"
              % (worst_mean, worst_max, max_foreign_cpu))
        if bad:
            fails.append(bad)

    if max_loadavg > 0:
        for r in rows:
            try:
                la = float(r.get("loadavg") or 0.0)
            except ValueError:
                fails.append("the load average column is not a number")
                break
            if la >= max_loadavg:
                fails.append("the load average %s on this row is not below "
                             "the quiet-window threshold %.2f -- the batch "
                             "was measured on a busy host"
                             % (la, max_loadavg))
                break

    return fails


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())

    def first(pred):
        for r in rows:
            if pred(r):
                return r
        raise SystemExit("the mutation %s has no row to act on" % kind)

    if kind == "drop_baseline":
        rows = [r for r in rows if r["variant"] != BASELINE]
    elif kind == "drop_variant":
        v = rows[0]["variant"]
        rows = [r for r in rows if r["variant"] != v]
    elif kind == "bad_verdict":
        first(lambda r: True)["verdict"] = "BADCODEC"
    elif kind == "zero_time":
        first(lambda r: True)["wall_s"] = "0"
    elif kind == "no_rss":
        first(lambda r: True)["rss_kb"] = "0"
    elif kind == "no_loadavg":
        first(lambda r: True)["loadavg"] = ""
    elif kind == "foreign_busy":
        # somebody else was using the host while this row was measured
        first(lambda r: True)["foreign_cpu"] = "4.000"
    elif kind == "no_foreign":
        first(lambda r: True)["foreign_cpu"] = ""
    elif kind == "no_provenance":
        first(lambda r: True)["commit"] = ""
    elif kind == "dup_row":
        rows = rows + [dict(rows[0])]
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["drop_baseline", "drop_variant", "bad_verdict", "zero_time",
             "no_rss", "no_loadavg", "foreign_busy", "no_foreign",
             "no_provenance", "dup_row"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="unified_ablation.csv")
    ap.add_argument("--max-loadavg", type=float, default=0.0,
                    help="fail if any row was measured at or above this load "
                         "average; 0 = do not check. The load average counts "
                         "OUR OWN threads, so it is a coarse diagnostic, not "
                         "the quiet-window gate -- that is --max-foreign-cpu.")
    ap.add_argument("--max-foreign-cpu", type=float, default=0.25,
                    help="fail if any row was measured while somebody else "
                         "was using more than this many cores; 0 = do not "
                         "check")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    fails = check(args.csv, max_loadavg=args.max_loadavg,
                  max_foreign_cpu=args.max_foreign_cpu)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: every variant has a baseline and enough repetitions, every "
          "run reproduced the reference barcode, and every row is identified")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="ablation_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, max_loadavg=args.max_loadavg)
                if f:
                    print("  self-test %-16s -> correctly FAILED (%d problem(s))"
                          % (kind, len(f)))
                else:
                    print("  self-test %-16s -> *** NOT DETECTED ***" % kind)
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
