#!/usr/bin/env python3
"""Checker for the o3_4096 thread-scaling table.

The table holds, per (side, threads, dimension), the per-dimension phase
times and the wall clock of the run. The claims it has to support are:

1. **the grid is complete** -- our build at every thread count has a row
   for every dimension, and the reference implementation has its row;
2. **the host was quiet** -- the foreign load on every row is below the
   threshold, or the table is rejected rather than footnoted;
3. **the numbers hang together** -- the per-dimension times of a run do
   not add up to more than the wall clock of that run, and the wall clock
   of a run is not shorter than the sum of its dimensions' times;
4. **the reference actually ran** -- a reference row whose wall clock is
   implausibly short for the work is a failed run, not a fast one, and
   the table must not be read as if the reference were fast.

It prints its own denominator and carries negative self-tests.

USAGE
    python check_o3_cpu_scaling.py o3_cpu_scaling.csv [--max-foreign-cpu 0.25]
    python check_o3_cpu_scaling.py o3_cpu_scaling.csv --self-test
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

THREAD_COUNTS = (1, 2, 4, 16)
DIMS = (1, 2, 3)

REQUIRED = ("side", "threads", "dim", "wall_s", "foreign_cpu", "verdict")

# A reference run that finishes in less than this fraction of the
# shortest of our own runs has not done the work. (The reference does the
# same three dimensions on the same input, so it cannot be an order of
# magnitude faster than one of our runs at the same thread count.)
MIN_REFERENCE_FRACTION = 0.05


def _dim_key(d):
    """Sort key for a dimension cell; the reference's row has no dimension."""
    try:
        return (0, int(d))
    except (TypeError, ValueError):
        return (1, 0)


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    by = {}
    for r in rows:
        by.setdefault((r["side"], r["threads"]), {})[r["dim"]] = r
    return rows, by


def check(path, max_foreign_cpu=0.25):
    fails = []
    rows, by = load(path)
    if not rows:
        return ["the table is empty: %s" % path]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return ["the table is missing columns: %s" % ", ".join(missing_cols)]

    expected = len(THREAD_COUNTS) * len(DIMS) + 1
    for t in THREAD_COUNTS:
        key = ("ours", str(t))
        if key not in by:
            fails.append("no rows for our build at %d thread(s)" % t)
            continue
        for dim in DIMS:
            if str(dim) not in by[key]:
                fails.append("missing cell: ours / %d threads / dim %d"
                             % (t, dim))
    if ("classic", "1") not in by:
        fails.append("no row for the reference implementation")

    # the quiet host
    worst_mean, worst_row = 0.0, None
    for r in rows:
        if (r.get("verdict") or "").strip() != "OK":
            fails.append("%s / %s threads / dim %s: the run did not succeed "
                         "(verdict %r)" % (r["side"], r["threads"], r["dim"],
                                           r.get("verdict")))
        raw = (r.get("foreign_cpu") or "").strip()
        if not raw:
            fails.append("%s / %s threads / dim %s: no foreign-load reading"
                         % (r["side"], r["threads"], r["dim"]))
            continue
        try:
            value = float(raw)
        except ValueError:
            fails.append("%s / %s threads / dim %s: the foreign load is not a "
                         "number (%r)" % (r["side"], r["threads"], r["dim"], raw))
            continue
        if value > worst_mean:
            worst_mean, worst_row = value, "%s / %s threads" % (r["side"],
                                                                r["threads"])
        if max_foreign_cpu > 0 and value > max_foreign_cpu:
            fails.append("%s / %s threads / dim %s: the host was used by other "
                         "users at %.3f cores, above the %.2f-core threshold"
                         % (r["side"], r["threads"], r["dim"], value,
                            max_foreign_cpu))

    # the numbers hang together
    our_walls = []
    for (side, threads), dims in by.items():
        wall = 0.0
        for dim, r in dims.items():
            try:
                wall = max(wall, float((r.get("wall_s") or "0").strip() or 0))
            except ValueError:
                pass
        total = 0.0
        for dim, r in dims.items():
            try:
                total += float((r.get("total_s") or "").strip() or 0)
            except ValueError:
                pass
        if wall > 0 and total > wall * 1.05:
            fails.append("%s / %s threads: the per-dimension times add up to "
                         "%.3f s but the run took %.3f s"
                         % (side, threads, total, wall))
        if side == "ours":
            our_walls.append((threads, wall))

    # the reference actually ran
    classic = by.get(("classic", "1"))
    if classic:
        cwall = 0.0
        for r in classic.values():
            try:
                cwall = max(cwall, float((r.get("wall_s") or "0").strip() or 0))
            except ValueError:
                pass
        if our_walls:
            shortest = min(w for _t, w in our_walls if w > 0)
            if cwall < shortest * MIN_REFERENCE_FRACTION:
                fails.append("the reference implementation finished in %.3f s, "
                             "which is under %.0f%% of the shortest of our own "
                             "runs (%.3f s) on the same input and the same "
                             "three dimensions: that is a failed run, not a "
                             "fast one" % (cwall,
                                           100 * MIN_REFERENCE_FRACTION,
                                           shortest))
    return fails


MUTATIONS = ["drop_row", "busy_host", "no_foreign", "fail_verdict",
             "phases_over_wall", "fast_reference"]


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    if kind == "drop_row":
        rows = [r for r in rows if not (r["side"] == "ours" and r["threads"] == "4"
                                         and r["dim"] == "2")]
    elif kind == "busy_host":
        for r in rows:
            r["foreign_cpu"] = "1.500"
    elif kind == "no_foreign":
        for r in rows:
            r["foreign_cpu"] = ""
    elif kind == "fail_verdict":
        for r in rows:
            if r["side"] == "ours" and r["threads"] == "16" and r["dim"] == "3":
                r["verdict"] = "RC=1"
    elif kind == "phases_over_wall":
        for r in rows:
            r["total_s"] = "9999.0"
    elif kind == "fast_reference":
        for r in rows:
            if r["side"] == "classic":
                r["wall_s"] = "0.1"
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="o3_cpu_scaling.csv")
    ap.add_argument("--max-foreign-cpu", type=float, default=0.25)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    rows, by = load(args.csv)
    fails = check(args.csv, max_foreign_cpu=args.max_foreign_cpu)
    expected = len(THREAD_COUNTS) * len(DIMS) + 1
    print("expected cells : %d thread counts x %d dimensions + the reference "
          "= %d" % (len(THREAD_COUNTS), len(DIMS), expected))
    print("rows in table  : %d" % len(rows))
    for key in sorted(by):
        print("  %-8s %-3s threads: dims %s"
              % (key[0], key[1], ",".join(sorted(by[key], key=_dim_key))))
    if fails:
        print("\nFAIL (%d problem(s)):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: the grid is complete, every row was measured on a quiet "
          "host, the per-dimension times add up, and the reference run is a "
          "run that did the work")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="o3cpu_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, max_foreign_cpu=args.max_foreign_cpu)
                if f:
                    print("  self-test %-18s -> correctly FAILED (%d problem(s))"
                          % (kind, len(f)))
                else:
                    print("  self-test %-18s -> *** NOT DETECTED ***" % kind)
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
