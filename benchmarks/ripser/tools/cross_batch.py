"""Do two independent measurements of the same thing agree?

A table measured on a shared box is only a measurement of the code if the
box was quiet while it was measured, and "the box was quiet" is hard to
assert after the fact. There is a second, independent line of evidence:
if the same table is measured twice, on a quiet box, the two copies agree,
and if a neighbour's job had been running, they would not. The two lines
of evidence are independent -- the first is the host accounting, the
second is the numbers themselves -- so a table that has both is much
harder to argue with.

This compares two or more scale-scan tables cell by cell and reports the
spread. The tolerance is a property of the MEASUREMENT, not of the
comparison: the reference implementation is not run-to-run reproducible on
this workload (its GPU reduction order varies, so a near-tie in the H2
classes resolves differently -- measured at 0.4% of the bar count), and
the host's own jitter is a few percent. So the default tolerance is 5%.

Usage:
    python cross_batch.py a.csv b.csv [--tolerance 0.05] [--out out.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# The arms and the column that is compared. `wall_s` is the process clock
# and it is the one the tables are written around, so it is what is
# compared; the library clock is compared too, where the row has one.
COMPARE = ("wall_s", "lib_wall_s")


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        out[(int(r["n"]), r["side"])] = r
    return out


def compare(paths, tolerance, columns=None):
    """(rows, problems). Each row is one cell and what the copies said."""
    columns = tuple(columns) if columns else COMPARE
    tables = [load(p) for p in paths]
    cells = set(tables[0])
    for t in tables[1:]:
        cells &= set(t)
    out, problems = [], []
    for key in sorted(cells):
        rec = {"n": key[0], "side": key[1], "copies": len(tables)}
        for col in columns:
            vals = []
            missing_in = []
            for i, t in enumerate(tables):
                raw = (t[key].get(col) or "").strip()
                if not raw:
                    missing_in.append(i + 1)
                    continue
                try:
                    vals.append(float(raw))
                except ValueError:
                    missing_in.append(i + 1)
            if not vals:
                # No copy has a number here. That is a result of its own --
                # the arm timed out in every copy -- and it is consistent.
                continue
            if missing_in:
                # SOME copy has a number and some do not. That is not a
                # result: it means a cell that was measured in one batch was
                # not measured in another, and the comparison would be
                # comparing a number against a hole. A blank is not a
                # success.
                problems.append("n=%d %s: %s has no value in copy %s but a "
                                "value in the other copy -- a missing number "
                                "is not a measurement"
                                % (key[0], key[1], col,
                                   ", ".join(str(m) for m in missing_in)))
                continue
            lo, hi = min(vals), max(vals)
            rec[col] = "%.4f" % hi
            rec[col + "_min"] = "%.4f" % lo
            rec[col + "_spread_pct"] = "%.3f" % (100.0 * (hi - lo) / hi
                                                 if hi else 0.0)
            if lo > 0 and (hi - lo) / hi > tolerance:
                problems.append("n=%d %s: %s ranges from %.4f to %.4f over "
                                "%d copies -- a spread of %.1f%%, above the "
                                "%.1f%% tolerance"
                                % (key[0], key[1], col, lo, hi, len(vals),
                                   100.0 * (hi - lo) / hi,
                                   100.0 * tolerance))
        out.append(rec)
    # A cell that is missing from one of the copies cannot be compared, and
    # saying so is better than quietly comparing fewer cells than there are.
    for i, t in enumerate(tables):
        for key in sorted(set(t) - cells):
            problems.append("n=%d %s: the cell is missing from copy %d, so it "
                            "cannot be compared" % (key[0], key[1], i + 1))
    return out, problems


def mutate(src, dst, kind):
    """Corrupt one copy, for the checker's own negative self-test."""
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    if not rows:
        raise SystemExit("the table is empty")
    if kind == "slow_one_copy":
        # one cell in one copy is 30% slower than the other: the signature
        # of a batch that was measured on a busy host
        for r in rows:
            if r.get("side") == "ours_inf":
                try:
                    r["wall_s"] = "%.4f" % (float(r["wall_s"]) * 1.3)
                except (ValueError, TypeError):
                    pass
                break
    elif kind == "no_number":
        rows[0]["wall_s"] = ""
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def self_test(path, tolerance):
    """Corrupt a copy of the table and check that the comparison notices."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="cross_batch_self_")
    try:
        bad = 0
        for kind in ("slow_one_copy", "no_number"):
            a = os.path.join(tmp, "a_%s.csv" % kind)
            b = os.path.join(tmp, "b_%s.csv" % kind)
            shutil.copy(path, a)
            mutate(a, b, kind)
            _rows, problems = compare([a, b], tolerance)
            if problems:
                print("  self-test %-16s -> correctly FAILED (%d problem(s))"
                      % (kind, len(problems)))
            else:
                print("  self-test %-16s -> *** NOT DETECTED ***" % kind)
                bad += 1
        if bad:
            print("\nFAIL: %d mutation(s) slipped through" % bad)
            return 1
        print("\nPASS: all 2 negative self-tests detected")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="the largest spread between copies that is still "
                         "called agreement (0.05 = 5%%)")
    ap.add_argument("--out", default="")
    ap.add_argument("--columns", default="",
                    help="comma-separated columns to compare; empty = all of "
                         "%s. Use this when the copies have different schemas "
                         "-- an older table that predates a column has no "
                         "value in it, which is a schema difference and not "
                         "a disagreement about the numbers." % ", ".join(COMPARE))
    ap.add_argument("--self-test", action="store_true",
                    help="corrupt a copy of the table and check that this "
                         "comparison notices")
    args = ap.parse_args()

    if args.self_test:
        # the self-test builds its own second copy, so it needs only one
        return self_test(args.csv[0], args.tolerance)
    if len(args.csv) < 2:
        print("need at least two tables to compare", file=sys.stderr)
        return 2
    cols = [c.strip() for c in args.columns.split(",") if c.strip()] or None
    rows, problems = compare(args.csv, args.tolerance, cols)
    print("compared %d cell(s) across %d copies, tolerance %.1f%%"
          % (len(rows), len(args.csv), 100.0 * args.tolerance))
    if rows:
        print("\n| n | arm | copies | wall (max) | wall (min) | spread | "
              "lib wall (max) | spread |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print("| %s | %s | %s | %s | %s | %s%% | %s | %s%% |"
                  % (r["n"], r["side"], r["copies"],
                     r.get("wall_s", ""), r.get("wall_s_min", ""),
                     r.get("wall_s_spread_pct", ""),
                     r.get("lib_wall_s", ""),
                     r.get("lib_wall_s_spread_pct", "")))
    if args.out:
        fields = ["n", "side", "copies", "wall_s", "wall_s_min",
                  "wall_s_spread_pct", "lib_wall_s", "lib_wall_s_min",
                  "lib_wall_s_spread_pct"]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    if problems:
        print("\nFAIL (%d problem(s)):" % len(problems))
        for p in problems[:40]:
            print("  - %s" % p)
        return 1
    print("\nPASS: every cell the copies have in common agrees within the "
          "tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
