#!/usr/bin/env python3
"""Checker for the T-arm equivalence table.

The table answers one question: does the reference's threshold-at-the-
maximum-finite-distance arm compute the SAME filtration as our keeping-
the-`+inf` representation -- and do their barcodes agree, value by value,
including the essential bars?

  1. THE GRID IS COMPLETE. Every size x every dimension.
  2. THE COUNTS LINE UP. The bars that die at or below T on our side must
     be the reference's finite bars, one for one; and our bars that die
     above T (truncated) plus our genuinely essential bars must be the
     reference's essential bars, one for one.
  3. THE VALUES AGREE. Every matched pair within the tolerance the
     reference's six printed digits allow (1e-5 relative), and the
     essential bars agree on their BIRTHS -- their deaths are infinite on
     both sides by construction, so only the births carry information.
  4. AND THE FINDING IS REPORTED: the number of bars whose death is
     truncated by T. For a distance-matrix filtration it is ZERO at every
     dimension, because every finite death is a filtration value of a
     simplex, hence one of the matrix's entries, hence at most T -- so
     thresholding at T cannot change any death. The checker prints the
     column rather than assuming it.

USAGE
    python check_t_arm_equivalence.py t_arm_equivalence.csv [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

REQUIRED = ["n", "dim", "T", "ours_dying", "ours_truncated", "ours_essential",
            "pp_finite", "pp_essential", "matched", "unmatched_ours",
            "unmatched_pp", "max_rel_dev", "birth_matched", "birth_unmatched",
            "verdict"]
RTOL = 1e-5


def check(path):
    fails = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["table is empty: %s" % path]
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        return ["table is missing columns: %s" % ", ".join(missing)]

    ns = sorted({int(r["n"]) for r in rows})
    dims = sorted({int(r["dim"]) for r in rows})
    print("sizes %s x dimensions %s = %d rows expected"
          % (ns, dims, len(ns) * len(dims)))
    print("rows in table                    : %d" % len(rows))

    by = {}
    for r in rows:
        key = (int(r["n"]), int(r["dim"]))
        if key in by:
            fails.append("duplicated row: n=%d dim=%d" % key)
            continue
        by[key] = r
    for n in ns:
        for d in dims:
            if (n, d) not in by:
                fails.append("missing cell: n=%d dim=%d" % (n, d))

    print("\n| n | dim | dying | truncated | essential | matched | max rel dev | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for n in ns:
        for d in dims:
            r = by.get((n, d))
            if not r:
                continue
            if r["verdict"] != "SAME_FILTRATION":
                fails.append("n=%d dim=%d: verdict %s" % (n, d, r["verdict"]))
            try:
                dying = int(r["ours_dying"])
                trunc = int(r["ours_truncated"])
                ess = int(r["ours_essential"])
                ppf = int(r["pp_finite"])
                ppe = int(r["pp_essential"])
                matched = int(r["matched"])
                uo, up = int(r["unmatched_ours"]), int(r["unmatched_pp"])
                bm, bu = int(r["birth_matched"]), int(r["birth_unmatched"])
                dev = float(r["max_rel_dev"])
            except ValueError:
                fails.append("n=%d dim=%d: the columns are not numbers" % (n, d))
                continue
            if dying != ppf:
                fails.append("n=%d dim=%d: we have %d bars dying at or below T "
                             "and the reference has %d finite bars -- the two "
                             "arms do not describe the same filtration"
                             % (n, d, dying, ppf))
            if trunc + ess != ppe:
                fails.append("n=%d dim=%d: our truncated+essential bars (%d+%d) "
                             "do not account for the reference's %d essential "
                             "bars" % (n, d, trunc, ess, ppe))
            if matched != dying or uo or up:
                fails.append("n=%d dim=%d: %d of %d bars matched, %d of ours "
                             "and %d of the reference's did not"
                             % (n, d, matched, dying, uo, up))
            if dev > RTOL:
                fails.append("n=%d dim=%d: the matched bars differ by %.3e, "
                             "more than the %.0e the reference's six printed "
                             "digits allow" % (n, d, dev, RTOL))
            if bm != trunc + ess or bu:
                fails.append("n=%d dim=%d: %d of the essential bars' births "
                             "matched, %d did not" % (n, d, bm, bu))
            print("| %d | %d | %d | %d | %d | %d | %.3e | %s |"
                  % (n, d, dying, trunc, ess, matched, dev, r["verdict"]))
    trunc_total = sum(int(by[k]["ours_truncated"]) for k in by)
    print("\nbars whose death the threshold truncates, over every cell: %d"
          % trunc_total)
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

    if kind == "drop_cell":
        rows = rows[1:]
    elif kind == "bad_verdict":
        first(lambda r: True)["verdict"] = "DIFFERENT"
    elif kind == "count_mismatch":
        first(lambda r: True)["pp_finite"] = "1"
    elif kind == "truncated_essential":
        first(lambda r: True)["ours_truncated"] = "7"
    elif kind == "unmatched":
        first(lambda r: True)["unmatched_ours"] = "3"
    elif kind == "deviation":
        first(lambda r: True)["max_rel_dev"] = "1.000e-02"
    elif kind == "birth_mismatch":
        first(lambda r: True)["birth_unmatched"] = "2"
    elif kind == "dup_row":
        rows = rows + [dict(rows[0])]
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["drop_cell", "bad_verdict", "count_mismatch",
             "truncated_essential", "unmatched", "deviation",
             "birth_mismatch", "dup_row"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="t_arm_equivalence.csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    fails = check(args.csv)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: the T arm and the `+inf` representation describe the same "
          "filtration at every cell, and their barcodes agree value by value")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="tarm_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst)
                if f:
                    print("  self-test %-20s -> correctly FAILED (%d problem(s))"
                          % (kind, len(f)))
                else:
                    print("  self-test %-20s -> *** NOT DETECTED ***" % kind)
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
