#!/usr/bin/env python3
"""Checker for the grid-cells main benchmark (the 485-cell grid).

The corpus's own driver (`~/gridcells/gca_bench.py`) writes one row per
(matrix, target, input_form) cell. This checker is the gate on that table:

  1. THE GRID IS COMPLETE. The denominator is computed HERE, from the
     matrix directory and the arm list -- never from the table's own row
     count. Every cell must appear exactly once, and a cell that is
     missing is named.
  2. THE ARMS OF A MATRIX AGREE. The four canns-lib arms (GPU and CPU,
     the +inf and the capped input) must produce the SAME barcode md5:
     the input form and the build must not change the answer.
  3. THE MANIFEST REFERENCE IS REPRODUCED. The corpus's own manifest
     carries the H1 top-3 lifetimes computed by the dataset authors with
     canns 1.0.2.dev / canns-lib 0.7.0. Our arms must reproduce them,
     and ripser++'s arm must produce the same number of bars.
  4. THE TABLE IS USABLE. Every row carries a positive wall time, a
     positive peak RSS and the machine's load average (the machine is
     shared, and a paper's numbers have to say what else was running).

USAGE
    python check_gridcells_bench.py gca_results_final.csv [--matrix-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

MATRIX_DIR = "/home/sichaohe/gridcells/matrices"

# THE ARM LIST IS NOT WRITTEN HERE. It lives in `harness/corpus_arms.py`,
# because the corpus' own driver reads it too -- and the driver is not in
# this repository, so a copy in each place was a coupling by hand. The
# driver imports from this repository (it already does, for foreign_cpu),
# so a move here used to break it silently.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from corpus_arms import ARMS

REQUIRED = ["matrix", "target", "input_form", "wall_s", "peak_rss_kb",
            "bars_d0", "bars_d1", "h1_top3", "h1_md5", "ref_h1_top3",
            "loadavg", "foreign_cpu", "foreign_max"]


def same_top3(a, b, rtol=1e-6):
    """The manifest writes the lifetimes with "/" and two decimals; the driver
    writes them with "|" and whatever precision the value had. Compare the
    NUMBERS, not the strings."""
    def parse(t):
        parts = [p for p in t.replace("|", "/").replace(",", "/").split("/") if p.strip()]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None
    pa, pb = parse(a), parse(b)
    if pa is None or pb is None or len(pa) != len(pb):
        return a.strip() == b.strip()
    for x, y in zip(pa, pb):
        if abs(x - y) > rtol * max(1.0, abs(x)):
            return False
    return True


def expected_matrices(matrix_dir=MATRIX_DIR):
    if not os.path.isdir(matrix_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(matrix_dir) if f.endswith(".npy"))


def check(path, matrix_dir=MATRIX_DIR, max_foreign_cpu=0.25):
    """Returns a list of failure strings (empty == PASS)."""
    fails = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["table is empty: %s" % path]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return ["table is missing columns: %s" % ", ".join(missing_cols)]

    mats = expected_matrices(matrix_dir)
    if not mats:
        fails.append("no matrices under %s -- this checker has to run where the "
                     "corpus is" % matrix_dir)
    want = len(mats) * len(ARMS)
    print("expected cells: %d matrices x %d arms = %d" % (len(mats), len(ARMS), want))
    print("rows in table                    : %d" % len(rows))

    seen = {}
    for r in rows:
        key = (r["matrix"], r["target"], r["input_form"])
        if key in seen:
            fails.append("duplicated cell: %s / %s / %s" % key)
        seen[key] = r

    for m in mats:
        for target, form in ARMS:
            r = seen.get((m, target, form))
            if r is None:
                fails.append("missing cell: %s / %s / %s" % (m, target, form))
                continue
            try:
                if float(r["wall_s"]) <= 0:
                    fails.append("%s/%s/%s: non-positive wall time" % (m, target, form))
                if int(r["peak_rss_kb"]) <= 0:
                    fails.append("%s/%s/%s: no peak RSS" % (m, target, form))
            except ValueError:
                fails.append("%s/%s/%s: the timing columns are not numbers"
                             % (m, target, form))
            if not (r.get("loadavg") or "").strip():
                fails.append("%s/%s/%s: no load average recorded (the machine is "
                             "shared)" % (m, target, form))
            if not (r.get("foreign_cpu") or "").strip():
                fails.append("%s/%s/%s: no foreign-load reading -- without it "
                             "the table cannot say whether it was measured on "
                             "a quiet host" % (m, target, form))

    # The quiet-window gate is the FOREIGN load: the number of cores used by
    # processes that are not ours while the cell was measured. The load
    # average cannot serve as the gate -- it counts our own threads, and the
    # ratio this table reports moves with it (measured: 0.617 at a load
    # average below 1, 0.674 at 3-4) -- but it cannot say whether the load was
    # ours or a neighbour's either. The foreign reading can.
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
                bad = ("%s/%s/%s: the foreign load %s cores is above the "
                       "quiet-window threshold %.2f -- somebody else was "
                       "using the host while this cell was measured"
                       % (r["matrix"], r["target"], r["input_form"], raw,
                          max_foreign_cpu))
        print("worst foreign load: mean %.3f cores, largest single window "
              "%.3f cores (threshold %.2f)"
              % (worst_mean, worst_max, max_foreign_cpu))
        if bad:
            fails.append(bad)

    # the arms of a matrix must agree on the barcode
    by_matrix = {}
    for r in rows:
        by_matrix.setdefault(r["matrix"], []).append(r)
    for m, rs in by_matrix.items():
        canns = [r for r in rs if r["target"].startswith("canns")]
        md5s = {r["h1_md5"] for r in canns if r.get("h1_md5")}
        if len(md5s) > 1:
            fails.append("%s: the canns-lib arms disagree on the barcode: %s"
                         % (m, sorted(md5s)))
        for r in canns:
            ref = (r.get("ref_h1_top3") or "").strip()
            got = (r.get("h1_top3") or "").strip()
            if ref and got and not same_top3(ref, got):
                fails.append("%s/%s/%s: the H1 top-3 %s does not reproduce the "
                             "manifest reference %s"
                             % (m, r["target"], r["input_form"], got, ref))
        pp = [r for r in rs if r["target"].startswith("ripserpp")]
        for r in pp:
            for arm in canns:
                if r.get("bars_d1") and arm.get("bars_d1") and \
                        r["bars_d1"] != arm["bars_d1"]:
                    fails.append("%s: %s reports %s H1 bars, canns-lib %s"
                                 % (m, r["target"], r["bars_d1"], arm["bars_d1"]))
    return fails


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    if kind == "drop_cell":
        rows = rows[:-1]
    elif kind == "dup_cell":
        rows = rows + [dict(rows[0])]
    elif kind == "disagree_arms":
        for r in rows:
            if r["target"] == "canns-cpu":
                r["h1_md5"] = "deadbeef" * 4
                break
    elif kind == "break_reference":
        # A row that actually HAS a manifest reference: the first matrix may
        # not have one, and then the mutation would be invisible.
        for r in rows:
            if (r.get("ref_h1_top3") or "").strip():
                r["h1_top3"] = "999.00,888.00,777.00"
                break
    elif kind == "zero_time":
        rows[0]["wall_s"] = "0"
    elif kind == "no_loadavg":
        rows[0]["loadavg"] = ""
    elif kind == "foreign_busy":
        # somebody else was using the host while this cell was measured
        rows[0]["foreign_cpu"] = "2.500"
    elif kind == "no_foreign":
        rows[0]["foreign_cpu"] = ""
    elif kind == "bars_disagree":
        for r in rows:
            if r["target"] == "ripserpp":
                r["bars_d1"] = "999"
                break
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["drop_cell", "dup_cell", "disagree_arms", "break_reference",
             "zero_time", "no_loadavg", "foreign_busy", "no_foreign",
             "bars_disagree"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="gca_results_final.csv")
    ap.add_argument("--matrix-dir", default=MATRIX_DIR)
    ap.add_argument("--max-foreign-cpu", type=float, default=0.25,
                    help="fail if any cell was measured while somebody else "
                         "was using more than this many cores; 0 = do not "
                         "check. The load average counts OUR OWN threads, so "
                         "it is a diagnostic, not the gate.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    fails = check(args.csv, args.matrix_dir, max_foreign_cpu=args.max_foreign_cpu)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        if len(fails) > 40:
            print("  ... and %d more" % (len(fails) - 40))
        return 1
    print("\nPASS: every cell of the grid is present, the arms agree, and the "
          "manifest reference is reproduced")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="gcbench_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, args.matrix_dir, max_foreign_cpu=args.max_foreign_cpu)
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
