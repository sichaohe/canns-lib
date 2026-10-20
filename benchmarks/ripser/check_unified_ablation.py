#!/usr/bin/env python3
"""Checker for the unified ablation table.

  1. THE GRID IS COMPLETE. Every change x every arm x every repetition
     must have BOTH a `before` and an `after` measurement. The denominator
     is computed here from the table's own contents, not remembered.
  2. EVERY RUN REPRODUCED THE REFERENCE BARCODE. An ablation row that
     changed the answer is not an ablation row, it is a bug.
  3. THE MEASUREMENTS ARE IDENTIFIED. The load average, the device and the
     source fingerprint are on every row, so a reader can tell what was
     measured and on what.
  4. THE EFFECT IS STATED WITH ITS NOISE. The checker prints the median
     of the `after` over the median of the `before`, and compares the
     effect with the spread of the `before` measurements themselves: an
     effect smaller than the measurement's own spread is reported as
     "within noise", not as a win.

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

REQUIRED = ["rep", "change", "arm", "role", "build", "wall_s", "rss_kb",
            "bars", "md5", "verdict", "loadavg", "gpu", "commit"]
ROLES = ("before", "after")


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def check(path, min_reps=3):
    fails = []
    rows = load(path)
    if not rows:
        return ["table is empty: %s" % path]
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        return ["table is missing columns: %s" % ", ".join(missing)]

    changes = sorted({r["change"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    reps = sorted({int(r["rep"]) for r in rows})
    print("changes %d x arms %d x repetitions %d x roles %d = %d rows expected"
          % (len(changes), len(arms), len(reps), len(ROLES),
             len(changes) * len(arms) * len(reps) * len(ROLES)))
    print("rows in table                    : %d" % len(rows))

    cells = {}
    for r in rows:
        key = (r["change"], r["arm"], int(r["rep"]), r["role"])
        if key in cells:
            fails.append("duplicated row: %s / %s / rep %s / %s"
                         % (r["change"], r["arm"], r["rep"], r["role"]))
            continue
        cells[key] = r

    for c in changes:
        for a in arms:
            for rep in reps:
                for role in ROLES:
                    if (c, a, rep, role) not in cells:
                        fails.append("missing cell: %s / %s / rep %d / %s"
                                     % (c, a, rep, role))

    for key, r in cells.items():
        if r["verdict"] != "OK":
            fails.append("%s / %s / rep %s / %s: verdict %s"
                         % (r["change"], r["arm"], r["rep"], r["role"],
                            r["verdict"]))
        for col in ("loadavg", "gpu", "commit"):
            if not (r.get(col) or "").strip():
                fails.append("%s / %s / rep %s / %s: the row does not record %s"
                             % (r["change"], r["arm"], r["rep"], r["role"], col))
        try:
            if float(r["wall_s"]) <= 0:
                fails.append("%s / %s / rep %s / %s: non-positive wall time"
                             % (r["change"], r["arm"], r["rep"], r["role"]))
            if int(r["rss_kb"]) <= 0:
                fails.append("%s / %s / rep %s / %s: no peak RSS"
                             % (r["change"], r["arm"], r["rep"], r["role"]))
        except ValueError:
            fails.append("%s / %s / rep %s / %s: the timing columns are not "
                         "numbers" % (r["change"], r["arm"], r["rep"], r["role"]))

    # --- the effect of each change, with its noise ------------------------
    print("\n| change | arm | before (s) | after (s) | time | peak RSS | memory | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for c in changes:
        for a in arms:
            bef = [float(cells[(c, a, rep, "before")]["wall_s"])
                   for rep in reps if (c, a, rep, "before") in cells]
            aft = [float(cells[(c, a, rep, "after")]["wall_s"])
                   for rep in reps if (c, a, rep, "after") in cells]
            if not bef or not aft:
                continue
            bm, am = statistics.median(bef), statistics.median(aft)
            spread = (max(bef) - min(bef)) / bm if bm else 0.0
            ratio = am / bm if bm else 0.0
            rb = [int(cells[(c, a, rep, "before")]["rss_kb"]) / 1048576.0
                  for rep in reps if (c, a, rep, "before") in cells]
            ra = [int(cells[(c, a, rep, "after")]["rss_kb"]) / 1048576.0
                  for rep in reps if (c, a, rep, "after") in cells]
            rmb, rma = statistics.median(rb), statistics.median(ra)
            # the effect is only a finding if it is bigger than the spread
            # of the measurements it is computed from
            verdict = ("within noise" if abs(ratio - 1.0) <= spread
                       else ("faster" if ratio < 1.0 else "slower"))
            print("| %s | %s | %.2f | %.2f | %.3fx | %.1f -> %.1f GB | %.3fx | %s |"
                  % (c, a, bm, am, ratio, rmb, rma,
                     (rma / rmb if rmb else 0.0), verdict))
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

    if kind == "drop_role":
        rows = [r for r in rows
                if not (r["role"] == "after" and r["change"] == rows[0]["change"])]
    elif kind == "bad_barcode":
        first(lambda r: r["role"] == "after")["verdict"] = "BADCODEC"
    elif kind == "flip_verdict":
        first(lambda r: r["role"] == "before")["verdict"] = "BADCODEC"
    elif kind == "zero_time":
        first(lambda r: r["role"] == "after")["wall_s"] = "0"
    elif kind == "no_rss":
        first(lambda r: r["role"] == "after")["rss_kb"] = "0"
    elif kind == "no_loadavg":
        first(lambda r: r["role"] == "after")["loadavg"] = ""
    elif kind == "dup_row":
        rows = rows + [dict(rows[0])]
    elif kind == "no_provenance":
        first(lambda r: r["role"] == "after")["commit"] = ""
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["drop_role", "bad_barcode", "flip_verdict", "zero_time",
             "no_rss", "no_loadavg", "dup_row", "no_provenance"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="unified_ablation.csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    fails = check(args.csv)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: every change has a before and an after for every arm and "
          "repetition, every run reproduced the reference barcode, and every "
          "row is identified")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="ablation_self_")
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
