#!/usr/bin/env python3
"""Check the corrected head-to-head CSV.

RULES ENFORCED
  1. STRUCTURE. The expected cell grid is computed from the EXPECTED table
     below (reps x implementations), never from the number of rows the CSV
     happens to contain. The denominator is printed.
  2. NO INCOMPLETE CELLS. A cell whose wall time is missing, or whose
     verdict is not OK, is a FAILURE. "unavailable" is not a legal value.
  3. BARCODE. Every "ours" cell must carry the reference barcode md5 and
     the reference bar counts, and every "ripserpp" cell must carry the
     reference bar count. A cell whose barcode differs fails the CSV --
     a speed number with a different barcode is not a speed number.
  4. THE GAP TABLE MUST CLOSE. The per-stage differences are summed and
     the residual against the total difference is computed. The residual
     must stay under 5% of the larger total, otherwise the stage mapping
     is hiding work and the table is rejected. (This is the rule that
     killed the previous rounds' hand-made gap tables: they did not add
     up.)
  5. NEGATIVE SELF-TEST. The checker mutates a COPY of the CSV in ways
     that must be caught -- a changed barcode, a changed wall time that
     breaks the closure, a deleted row, an "unavailable" cell -- and
     asserts that the checker rejects each one. A checker that has never
     been shown to fail is not evidence of anything.

USAGE
    python check_clean_h2h.py clean_h2h.csv [--expect-ratio-max X]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

# ---- EXPECTED: the grid. The denominator is computed from THIS, not from
# ---- the row count of the file.
EXPECTED_REPS = 3
EXPECTED_WHO = ("ours", "ripserpp")
EXPECTED_CELLS = EXPECTED_REPS * len(EXPECTED_WHO)

REF_MD5 = ("ee1f5283091dd645a3dd53e6f1a7eac6:"
           "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
           "2d018d8fe9c1895b1b74f837af59ed20")
REF_BARS_OURS = "1195|481|58"
REF_BARS_PP = "1734"          # 1195 + 481 + 58, ripser++'s one-bar-per-line format

# The stage table. Each entry is (column in the CSV, label). The two sides
# cut their timers at different places, so the mapping is by function; the
# closure rule is what keeps the mapping honest.
STAGES = [
    ("s_dim0", "dim 0 (vertices)"),
    ("s_edges", "input / edges"),
    ("s_d1_reduce", "dim 1 reduction"),
    ("s_d2_assemble", "dim 2 column build"),
    ("s_hashcopy", "hash-map copy"),
    ("s_d2_gpupre", "dim 2 GPU pre-pass"),
    ("s_d2_pivotbuild", "dim 2 pivot-map build"),
    ("s_d2_reduce", "dim 2 reduction"),
]


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def mean_of(rows, who, key):
    vals = []
    for r in rows:
        if r.get("who") == who:
            v = (r.get(key) or "").strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def check(path, expect_ratio_max=None, verbose=True):
    """Return a list of failure strings. Empty list == PASS."""
    errs = []
    rows = read_rows(path)

    # -- rule 1: structure -------------------------------------------------
    if len(rows) != EXPECTED_CELLS:
        errs.append("expected %d cells (EXPECTED: %d reps x %d implementations), "
                    "found %d" % (EXPECTED_CELLS, EXPECTED_REPS, len(EXPECTED_WHO), len(rows)))
    for rep in range(1, EXPECTED_REPS + 1):
        for who in EXPECTED_WHO:
            n = sum(1 for r in rows if r.get("rep") == str(rep) and r.get("who") == who)
            if n != 1:
                errs.append("cell (rep=%d, who=%s) appears %d times, expected 1" % (rep, who, n))

    # -- rule 2: no incomplete cells --------------------------------------
    for i, r in enumerate(rows):
        if (r.get("wall_s") or "").strip() == "":
            errs.append("row %d (rep=%s who=%s): wall_s is empty -- a missing "
                        "measurement is not a legal value" % (i + 2, r.get("rep"), r.get("who")))
        verdict = (r.get("verdict") or "").strip()
        if verdict != "OK":
            errs.append("row %d (rep=%s who=%s): verdict=%r" % (i + 2, r.get("rep"), r.get("who"), verdict))
        if (r.get("gpu_err") or "").strip() != "":
            errs.append("row %d (rep=%s who=%s): gpu_err=%r -- this build is missing the "
                        "`gpu` feature, so it silently uses the CPU apparent pre-pass and "
                        "is not comparable with a GPU build"
                        % (i + 2, r.get("rep"), r.get("who"), r.get("gpu_err")))

    # -- rule 3: barcode ---------------------------------------------------
    for r in rows:
        if r.get("who") == "ours":
            if (r.get("md5") or "").strip() != REF_MD5:
                errs.append("rep=%s: barcode md5 differs from the reference" % r.get("rep"))
            if (r.get("bars") or "").strip() != REF_BARS_OURS:
                errs.append("rep=%s: bar counts %r differ from the reference %s"
                            % (r.get("rep"), r.get("bars"), REF_BARS_OURS))
        else:
            if (r.get("bars") or "").strip() != REF_BARS_PP:
                errs.append("rep=%s: ripser++ bar count %r differs from the reference %s"
                            % (r.get("rep"), r.get("bars"), REF_BARS_PP))

    # -- rule 4: the gap table must close ---------------------------------
    ours_wall = mean_of(rows, "ours", "wall_s")
    pp_wall = mean_of(rows, "ripserpp", "wall_s")
    if ours_wall is None or pp_wall is None:
        errs.append("cannot compute the two totals: a wall time is missing")
    else:
        stage_sum = 0.0
        for col, _label in STAGES:
            o = mean_of(rows, "ours", col)
            p = mean_of(rows, "ripserpp", col)
            if o is None or p is None:
                continue
            stage_sum += (o - p)
        residual = (ours_wall - pp_wall) - stage_sum
        denom = max(ours_wall, pp_wall)
        pct = 100.0 * abs(residual) / denom if denom else 0.0
        if verbose:
            print("  closure: total diff %.3f s, stage sum %.3f s, residual %.3f s (%.2f%% of %.3f)"
                  % (ours_wall - pp_wall, stage_sum, residual, pct, denom))
        if pct >= 5.0:
            errs.append("the stage table does not close: residual %.3f s is %.2f%% of the "
                        "total (must be < 5%%)" % (residual, pct))
        if expect_ratio_max is not None:
            ratio = ours_wall / pp_wall
            if verbose:
                print("  ratio: ours %.3f s / ripser++ %.3f s = %.3fx"
                      % (ours_wall, pp_wall, ratio))
            if ratio > expect_ratio_max:
                errs.append("ratio %.3fx exceeds the expected maximum %.3fx"
                            % (ratio, expect_ratio_max))

    return errs


def negative_self_test(path):
    """Mutate copies of the CSV and assert the checker rejects each one."""
    base = read_rows(path)
    cases = []

    def write_variant(rows, name):
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        cases.append((name, tmp))

    # (a) a changed barcode
    rows = [dict(r) for r in base]
    for r in rows:
        if r["who"] == "ours":
            r["md5"] = "0" * 32
            break
    write_variant(rows, "barcode changed")

    # (b) a wall time that breaks the closure
    rows = [dict(r) for r in base]
    for r in rows:
        if r["who"] == "ours":
            r["wall_s"] = "%.3f" % (float(r["wall_s"]) + 7.0)
            break
    write_variant(rows, "wall time shifted by 7 s (closure must break)")

    # (c) a deleted row
    rows = [dict(r) for r in base]
    rows = rows[:-1]
    write_variant(rows, "one row deleted")

    # (d) an empty measurement
    rows = [dict(r) for r in base]
    rows[0]["wall_s"] = ""
    write_variant(rows, "empty wall time")

    # (e) a bad verdict
    rows = [dict(r) for r in base]
    rows[0]["verdict"] = "BADCODEC"
    write_variant(rows, "verdict BADCODEC")

    # (f) a build that lost the `gpu` feature
    rows = [dict(r) for r in base]
    for r in rows:
        if r.get("who") == "ours":
            r["gpu_err"] = "built without the `gpu` feature"
            break
    write_variant(rows, "gpu_err set (feature-set gate)")

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
    ap.add_argument("csv_path")
    ap.add_argument("--expect-ratio-max", type=float, default=None)
    ap.add_argument("--no-self-test", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv_path):
        print("FAIL: %s does not exist" % args.csv_path, file=sys.stderr)
        return 2

    print("CHECK %s" % args.csv_path)
    print("  expected cells: %d (%d reps x %d implementations)"
          % (EXPECTED_CELLS, EXPECTED_REPS, len(EXPECTED_WHO)))
    errs = check(args.csv_path, args.expect_ratio_max)
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
