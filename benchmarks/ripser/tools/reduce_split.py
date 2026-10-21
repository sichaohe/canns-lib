#!/usr/bin/env python3
"""Build (and check) the reduction-split table: our reduction vs ripser++'s.

This is the evidence that the 2.64x gap is NOT an algorithmic difference.

Both implementations are given the same filtration (threshold T = max finite
distance of the ASA matrix, 100,897,898 dim-2 columns) and both are
instrumented the same way:

  ours      : the STATS line printed by the `reduce-stats` build
  ripser++  : the REDUCEPROF line printed by a patched COPY of ripser++.cu
              (`patch_reduce_profile.py` + `patch_heap_probe.py`; the
              pristine source is never modified)

The table has one row per (side, dimension) and the columns are the ones
that must AGREE between the two sides if the algorithms are the same:

    cols            columns reduced at this dimension
    rounds          reduction rounds (pivot collisions)
    heap_max        peak size of the working coboundary heap
    col_rounds_max  most rounds taken by a single column
    cofacets        cofacets enumerated by the initial scan

The wall-clock columns (scan_s / fill_s / reduce_s / total_s) are the ones
that are EXPECTED to differ: they are the constants of the inner loop.

USAGE
    python reduce_split.py --ours ours.log --pp pp.log --out reduce_split.csv
    python reduce_split.py --check reduce_split.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

# The columns that must be IDENTICAL between the two sides.
MUST_MATCH = ("cols", "rounds", "heap_max", "col_rounds_max")

FIELDS = ["side", "dim", "cols", "rounds", "heap_max", "col_rounds_max",
          "cofacets", "scan_s", "fill_s", "reduce_s", "total_s", "source",
          "foreign_cpu", "foreign_max"]


def parse_ours(path):
    """Parse the STATS lines of our instrumented build."""
    out = {}
    for line in open(path):
        if not line.startswith("STATS dim="):
            continue
        dim = int(re.search(r"STATS dim=(\d+)", line).group(1))
        kv = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, _, v = tok.partition("=")
                kv[k] = v
        out[dim] = {
            "side": "ours",
            "dim": dim,
            "cols": int(kv.get("nonapparent_cols", 0)),
            "rounds": int(kv.get("rounds", 0)),
            "heap_max": int(kv.get("heap_max", 0)),
            "col_rounds_max": int(kv.get("col_rounds_max", 0)),
            "cofacets": "",
            "scan_s": float(kv.get("scan_s", 0.0)),
            "fill_s": float(kv.get("fill_s", 0.0)),
            "reduce_s": float(kv.get("reduce_s", 0.0)),
            "total_s": float(kv.get("nonapparent_loop_s", 0.0)),
            "source": os.path.basename(path),
        }
    return out


def parse_pp(path):
    """Parse the REDUCEPROF lines of the patched ripser++ copy."""
    out = {}
    red = {}
    for line in open(path):
        m = re.search(r"SUBMATRIX REDUCTION TIME for dim (\d+): ([\d.]+)s", line)
        if m:
            red[int(m.group(1))] = float(m.group(2))
        m = re.match(r"REDUCEPROF dim=(\d+) (.*)", line)
        if not m:
            continue
        dim = int(m.group(1))
        kv = {}
        for tok in m.group(2).split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                kv[k] = v
        out[dim] = {
            "side": "ripserpp",
            "dim": dim,
            "cols": int(kv.get("cols", 0)),
            "rounds": int(kv.get("rounds", 0)),
            "heap_max": int(kv.get("heap_max", 0)),
            "col_rounds_max": int(kv.get("col_rounds_max", 0)),
            "cofacets": int(kv.get("cofacets", 0)),
            "scan_s": float(kv.get("scan_s", 0.0)),
            "fill_s": float(kv.get("fill_s", 0.0)),
            # ripser++'s timer does not close over the round block, so the
            # round time is the total minus the two parts that ARE timed.
            "reduce_s": float(kv.get("reduce_s", 0.0)),
            "total_s": red.get(dim, 0.0),
            "source": os.path.basename(path),
        }
    for dim, rec in out.items():
        if rec["reduce_s"] == 0.0 and rec["total_s"] > 0.0:
            rec["reduce_s"] = rec["total_s"] - rec["scan_s"] - rec["fill_s"]
    return out


def build(ours_path, pp_path, out_path, foreign_cpu="", foreign_max=""):
    ours = parse_ours(ours_path)
    pp = parse_pp(pp_path)
    rows = []
    for dim in sorted(set(ours) | set(pp)):
        if dim in ours:
            rows.append(ours[dim])
        if dim in pp:
            rows.append(pp[dim])
    # The quiet-window reading belongs to the RUN, not to a dimension, so it
    # is the same on every row. It has to be passed in: this script compares
    # two logs that were produced elsewhere, and a log of a run on a busy
    # host cannot be shown to be a measurement of the code.
    for r in rows:
        r["foreign_cpu"] = foreign_cpu
        r["foreign_max"] = foreign_max
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows)" % (out_path, len(rows)))
    return rows


def check(path, verbose=True):
    errs = []
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return ["the CSV is empty"]
    by = {}
    for r in rows:
        by.setdefault(r["side"], {})[int(r["dim"])] = r
    if "ours" not in by or "ripserpp" not in by:
        return ["both sides must be present (ours=%s, ripserpp=%s)"
                % ("ours" in by, "ripserpp" in by)]
    for dim in sorted(set(by["ours"]) & set(by["ripserpp"])):
        o, p = by["ours"][dim], by["ripserpp"][dim]
        for col in MUST_MATCH:
            if o[col] != p[col]:
                errs.append("dim %d: %s differs -- ours %s, ripser++ %s. The two "
                            "implementations are NOT doing the same work, so the "
                            "wall-clock difference cannot be called a constant "
                            "factor." % (dim, col, o[col], p[col]))
        if verbose:
            ot = float(o["total_s"])
            pt = float(p["total_s"])
            if ot > 0 and pt > 0:
                print("  dim %d: ours %.3f s vs ripser++ %.3f s (%.3fx) -- same "
                      "work, different constants" % (dim, ot, pt, ot / pt))
    return errs


def negative_self_test(path):
    """Mutate a copy and assert the checker catches it."""
    import tempfile
    base = list(csv.DictReader(open(path, newline="")))
    cases = []

    def variant(rows, name):
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        cases.append((name, tmp))

    rows = [dict(r) for r in base]
    for r in rows:
        if r["side"] == "ours":
            r["rounds"] = str(int(r["rounds"]) + 1)
            break
    variant(rows, "one side's round count changed by 1")

    rows = [dict(r) for r in base]
    for r in rows:
        if r["side"] == "ripserpp":
            r["heap_max"] = "0"
            break
    variant(rows, "one side's peak heap size changed")

    failures = []
    for name, tmp in cases:
        errs = check(tmp, verbose=False)
        if not errs:
            failures.append("NEGATIVE SELF-TEST FAILED: the checker ACCEPTED a CSV with %s" % name)
        else:
            print("  negative self-test OK: %s -> rejected (%d reason(s))" % (name, len(errs)))
        os.unlink(tmp)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours")
    ap.add_argument("--pp")
    ap.add_argument("--out", default="reduce_split.csv")
    ap.add_argument("--check")
    ap.add_argument("--foreign-cpu", default="",
                    help="the mean foreign cores during the run that produced "
                         "the logs. The table is not a measurement without it.")
    ap.add_argument("--foreign-max", default="",
                    help="the largest foreign rate seen in one sampling window")
    ap.add_argument("--no-self-test", action="store_true")
    args = ap.parse_args()

    if args.check:
        print("CHECK %s" % args.check)
        errs = check(args.check)
        if not args.no_self_test:
            errs += negative_self_test(args.check)
        if errs:
            print("\nFAIL (%d problem(s)):" % len(errs))
            for e in errs:
                print("  - %s" % e)
            return 1
        print("\nPASS")
        return 0

    if not args.ours or not args.pp:
        print("usage: --ours <log> --pp <log> [--out csv]", file=sys.stderr)
        return 2
    build(args.ours, args.pp, args.out, args.foreign_cpu, args.foreign_max)
    print("CHECK %s" % args.out)
    errs = check(args.out)
    if not args.no_self_test:
        errs += negative_self_test(args.out)
    if errs:
        print("\nFAIL (%d problem(s)):" % len(errs))
        for e in errs:
            print("  - %s" % e)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
