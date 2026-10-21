#!/usr/bin/env python3
"""Check the two-filtration sweep CSV.

The rules:
  1. GRID. The expected cell grid is computed from the EXPECTED table
     below (reps x arms), and the denominator is printed.
  2. NO INCOMPLETE CELLS. An empty wall time or a verdict other than OK
     is a failure.
  3. BARCODE. Every "ours" cell must carry the reference md5 and bar
     counts; every "ripserpp" cell must carry 1734 bars.
  4. THE ARMS MUST AGREE. `inf_T` and `cap_T` compute the same filtration
     and `inf_CAP` and `cap_CAP` compute the same one, so the two arms of
     each filtration MUST produce the same barcode. If they do not, the
     two input forms are not equivalent and the whole table is void --
     this is the check that would have caught the original mistake.
  5. NEGATIVE SELF-TEST. Copies of the CSV with a changed barcode, a
     changed bar count, a deleted row and an emptied wall time must all
     be rejected.

USAGE
    python check_filtration_sweep.py filtration_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import tempfile

import quiet_gate

EXPECTED_REPS = 2
EXPECTED_ARMS = ("inf_T", "inf_CAP", "cap_T", "cap_CAP", "T", "CAP")

REF_MD5 = ("ee1f5283091dd645a3dd53e6f1a7eac6:"
           "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
           "2d018d8fe9c1895b1b74f837af59ed20")
REF_BARS_OURS = "1195|481|58"
REF_BARS_PP = "1734"

# arms that must compute the same filtration
PAIRS = [("inf_T", "cap_T"), ("inf_CAP", "cap_CAP")]


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def mean_of(rows, side, arm):
    vals = []
    for r in rows:
        if r.get("side") == side and r.get("arm") == arm:
            v = (r.get("wall_s") or "").strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    return statistics.mean(vals) if vals else None


def check(path, verbose=True):
    errs = []
    rows = read(path)
    arms = sorted({r.get("arm") for r in rows})
    reps = sorted({r.get("rep") for r in rows})
    if verbose:
        print("  grid: %d reps x %d arms = %d cells"
              % (len(reps), len(arms), len(reps) * len(arms)))
    for rep in reps:
        for arm in EXPECTED_ARMS:
            n = sum(1 for r in rows if r.get("rep") == rep and r.get("arm") == arm)
            if n != 1:
                errs.append("cell (rep=%s arm=%s) appears %d times" % (rep, arm, n))
    for i, r in enumerate(rows):
        if (r.get("wall_s") or "").strip() == "":
            errs.append("row %d (rep=%s arm=%s): empty wall_s" % (i + 2, r.get("rep"), r.get("arm")))
        if (r.get("verdict") or "").strip() != "OK":
            errs.append("row %d (rep=%s arm=%s): verdict=%r"
                        % (i + 2, r.get("rep"), r.get("arm"), r.get("verdict")))
        if r.get("side") == "ours":
            if (r.get("md5") or "").strip() != REF_MD5:
                errs.append("row %d: barcode md5 differs from the reference" % (i + 2))
            if (r.get("bars") or "").strip() != REF_BARS_OURS:
                errs.append("row %d: bar counts %r differ from the reference %s"
                            % (i + 2, r.get("bars"), REF_BARS_OURS))
        else:
            if (r.get("bars") or "").strip() != REF_BARS_PP:
                errs.append("row %d: ripser++ bar count %r differs from the reference %s"
                            % (i + 2, r.get("bars"), REF_BARS_PP))

    # -- rule 4: the paired arms must agree
    for a, b in PAIRS:
        ba = {r.get("bars") for r in rows if r.get("arm") == a}
        bb = {r.get("bars") for r in rows if r.get("arm") == b}
        if ba and bb and ba != bb:
            errs.append("arms %s and %s compute the same filtration but disagree on "
                        "the barcode: %r vs %r" % (a, b, sorted(ba), sorted(bb)))

    if verbose:
        for filt, pp_arm, ours_arms in (("T", "T", ("inf_T", "cap_T")),
                                        ("CAP", "CAP", ("inf_CAP", "cap_CAP"))):
            p = mean_of(rows, "ripserpp", pp_arm)
            for oa in ours_arms:
                o = mean_of(rows, "ours", oa)
                if o and p:
                    print("  %-4s %-8s %.3f s  vs ripser++ %.3f s  = %.3fx"
                          % (filt, oa, o, p, o / p))
    return errs


def negative_self_test(path):
    base = read(path)
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
            r["md5"] = "0" * 32
            break
    variant(rows, "barcode md5 changed")

    rows = [dict(r) for r in base]
    for r in rows:
        if r["arm"] == "cap_T":
            r["bars"] = "1194|481|58"
            break
    variant(rows, "one arm of a pair disagrees on the bar count")

    rows = [dict(r) for r in base]
    rows[0]["wall_s"] = ""
    variant(rows, "empty wall time")

    rows = [dict(r) for r in base][:-1]
    variant(rows, "one row deleted")

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
    quiet_gate.add_arguments(ap)
    ap.add_argument("csv_path")
    ap.add_argument("--no-self-test", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv_path):
        print("FAIL: %s does not exist" % args.csv_path, file=sys.stderr)
        return 2
    print("CHECK %s" % args.csv_path)
    errs = check(args.csv_path)
    errs += quiet_gate.gate(args.csv_path, args.max_foreign_cpu,
                            label='filtration_sweep.csv')
    if not args.no_self_test:
        errs += negative_self_test(args.csv_path)
    if errs:
        print("\nFAIL (%d problem(s)):" % len(errs))
        for e in errs:
            print("  - %s" % e)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
