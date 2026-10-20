#!/usr/bin/env python3
"""Checker for the contention sweep.

  1. THE GRID IS COMPLETE. Every level x every side x every repetition
     is present, and the level-0 rows exist for every side -- without
     them there is no baseline to degrade from.
  2. EVERY RUN REPRODUCED THE BARCODE. A run that produced a different
     barcode is not a measurement of contention.
  3. THE DEGRADATION IS STATED PER SIDE. The checker prints, for each
     side, the median time at each level against the median at level 0.
     The two sides degrade differently, and that difference is the result.

USAGE
    python check_contention_sweep.py contention_sweep.csv [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import sys
import tempfile

REQUIRED = ["pass", "level", "side", "rep", "wall_s", "bars", "loadavg",
            "foreign_cpu", "foreign_max", "gpu",
            "verdict"]
SIDES = ("ours", "ripserpp")


def check(path):
    fails = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["table is empty: %s" % path]
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        return ["table is missing columns: %s" % ", ".join(missing)]

    levels = sorted({int(r["level"]) for r in rows})
    reps = sorted({int(r["rep"]) for r in rows})
    passes = sorted({int(r["pass"]) for r in rows})
    # the sweep goes UP and BACK DOWN, so the peak level is visited once and
    # the others twice: the grid is the set of (pass, level) pairs that
    # exist, not the product of the levels with the passes
    cells = sorted({(int(r["pass"]), int(r["level"])) for r in rows})
    print("levels %s, passes %s, repetitions %d" % (levels, passes, len(reps)))
    print("(pass, level) cells: %s" % cells)
    print("rows expected: %d cells x %d sides x %d repetitions = %d"
          % (len(cells), len(SIDES), len(reps),
             len(cells) * len(SIDES) * len(reps)))
    print("rows in table                    : %d" % len(rows))

    by = {}
    seen = set()
    for r in rows:
        key = (int(r["pass"]), int(r["level"]), r["side"], int(r["rep"]))
        if key in seen:
            fails.append("duplicated row: pass %d level %d %s rep %d" % key)
            continue
        seen.add(key)
        by.setdefault((int(r["level"]), r["side"]), []).append(r)
    # every (pass, level) cell must have both sides, with the same
    # repetitions -- otherwise one side was measured somewhere the other
    # was not, and the comparison at that level is not a comparison
    per_side_cells = {}
    for side in SIDES:
        per_side_cells[side] = sorted(
            {(int(r["pass"]), int(r["level"])) for r in rows if r["side"] == side})
    if per_side_cells[SIDES[0]] != per_side_cells[SIDES[1]]:
        fails.append("the two sides were not measured in the same cells: "
                     "%s against %s" % (per_side_cells[SIDES[0]],
                                        per_side_cells[SIDES[1]]))
    for (p, lv) in cells:
        for side in SIDES:
            got = [x for x in by.get((lv, side), []) if int(x["pass"]) == p]
            if len(got) != len(reps):
                fails.append("pass %d level %d / %s: %d repetition(s), "
                             "expected %d" % (p, lv, side, len(got), len(reps)))
    for side in SIDES:
        if not by.get((0, side)):
            fails.append("no baseline: level 0 / %s is missing, and without "
                         "it there is nothing to degrade from" % side)

    for r in rows:
        if r["verdict"] != "OK":
            fails.append("level %s / %s / rep %s: verdict %s"
                         % (r["level"], r["side"], r["rep"], r["verdict"]))
        if not (r.get("loadavg") or "").strip():
            fails.append("level %s / %s / rep %s: no load average recorded"
                         % (r["level"], r["side"], r["rep"]))
        if not (r.get("foreign_cpu") or "").strip():
            fails.append("level %s / %s / rep %s: no foreign-load reading -- "
                         "the competitors this experiment starts are ours, so "
                         "this is the interference we did NOT ask for"
                         % (r["level"], r["side"], r["rep"]))
        else:
            try:
                fc = float(r["foreign_cpu"])
            except ValueError:
                fc = None
            if fc is not None and fc > 0.25:
                fails.append("level %s / %s / rep %s: the foreign load %s cores "
                             "is above the quiet-window threshold 0.25"
                             % (r["level"], r["side"], r["rep"], r["foreign_cpu"]))
        try:
            if float(r["wall_s"]) <= 0:
                fails.append("level %s / %s / rep %s: non-positive wall time"
                             % (r["level"], r["side"], r["rep"]))
        except ValueError:
            fails.append("level %s / %s / rep %s: the wall time is not a number"
                         % (r["level"], r["side"], r["rep"]))

    # --- the degradation curve -------------------------------------------
    print("\n| level | competing processes | ours (s) | ours vs level 0 | "
          "ripser++ (s) | ripser++ vs level 0 |")
    print("|---|---|---|---|---|---|")
    for lv in levels:
        ours = [float(r["wall_s"]) for r in by.get((lv, "ours"), [])]
        ref = [float(r["wall_s"]) for r in by.get((lv, "ripserpp"), [])]
        if not ours or not ref:
            continue
        o0 = [float(r["wall_s"]) for r in by.get((0, "ours"), [])]
        r0 = [float(r["wall_s"]) for r in by.get((0, "ripserpp"), [])]
        om, rm = statistics.median(ours), statistics.median(ref)
        o0m = statistics.median(o0) if o0 else 0.0
        r0m = statistics.median(r0) if r0 else 0.0
        print("| %d | %d | %.2f | %s | %.2f | %s |"
              % (lv, lv, om, "%.2fx" % (om / o0m) if o0m else "-",
                 rm, "%.2fx" % (rm / r0m) if r0m else "-"))
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
        rows = [r for r in rows if r["level"] != "0"]
    elif kind == "drop_level":
        rows = [r for r in rows if r["level"] != rows[0]["level"]]
    elif kind == "bad_verdict":
        first(lambda r: True)["verdict"] = "BADCODE"
    elif kind == "zero_time":
        first(lambda r: True)["wall_s"] = "0"
    elif kind == "no_loadavg":
        first(lambda r: True)["loadavg"] = ""
    elif kind == "foreign_busy":
        first(lambda r: True)["foreign_cpu"] = "1.750"
    elif kind == "no_foreign":
        first(lambda r: True)["foreign_cpu"] = ""
    elif kind == "dup_row":
        rows = rows + [dict(rows[0])]
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["drop_baseline", "drop_level", "bad_verdict", "zero_time",
             "no_loadavg", "foreign_busy", "no_foreign", "dup_row"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="contention_sweep.csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    fails = check(args.csv)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: every level has both sides, the barcode was reproduced "
          "everywhere, and the baseline exists")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="contention_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst)
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
