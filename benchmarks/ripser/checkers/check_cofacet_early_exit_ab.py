#!/usr/bin/env python3
"""Checker for the cofacet early-exit A/B table.

The table holds, per (dataset, arm, repetition): the wall clock, the
per-dimension phase breakdown, the barcode, and the foreign load on the host.

**THE BARCODE MUST BE IDENTICAL IN EVERY ROW OF A DATASET.** The early exit
changes how many distances are READ, not which simplices are found. A row
whose barcode differs from the others is not a faster arm, it is a different
algorithm, and this checker rejects the table rather than reporting a
speed-up.

**THE REDUCTION ROUNDS MUST BE IDENTICAL ACROSS THE ARMS.** Same reason: the
switch must not change the amount of work, only its cost.

**BOTH ARMS MUST HAVE RUN THE SAME BINARY.** The switch is an environment
variable, so a table whose arms came from different builds says nothing about
the switch. The library path is compared row by row.

**THE EARLY-ON ARM MUST NOT BE SLOWER.** The change exists to make the
reduction cheaper; an arm that is more than 2% slower is a regression, and the
2% margin is the same one the rest of this repository uses for "must not
regress" claims.

**THE FOREIGN LOAD MUST BE UNDER THE GATE.** A row measured while somebody
else was using the machine is a row, but it is not a result. The gate is the
mean foreign cores, 0.25, and the peak is only a diagnostic.

It carries negative self-tests: a copy of the table with one thing changed
must be rejected.

USAGE
    python check_cofacet_early_exit_ab.py cofacet_early_exit_ab.csv
    python check_cofacet_early_exit_ab.py cofacet_early_exit_ab.csv --self-test
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

import quiet_gate

ARMS = ("early_on", "early_off")
# How much slower the early-on arm may be and still pass.
REGRESSION_MARGIN = 1.02

REQUIRED = ("dataset", "arm", "rep", "wall_s", "bars", "barhash", "foreign_cpu",
            "verdict", "lib")


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def check(path, verbose=True, max_foreign_cpu=quiet_gate.DEFAULT_GATE):
    """Returns a list of failure strings (empty == PASS)."""
    errs = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["the table is empty: %s" % path]
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        return ["the table is missing columns: %s" % ", ".join(missing)]

    for r in rows:
        if r["verdict"] != "OK":
            errs.append("%s/%s rep %s did not finish: verdict %r"
                        % (r["dataset"], r["arm"], r["rep"], r["verdict"]))
        if not r["bars"]:
            errs.append("%s/%s rep %s has no barcode" % (r["dataset"], r["arm"], r["rep"]))
    # The quiet-window gate is the shared one, so that "the numbers in this
    # table were measured on a busy host" is a FAILURE like everywhere else.
    errs += quiet_gate.gate(rows, max_foreign_cpu,
                             label="cofacet_early_exit_ab.csv")

    if errs:
        return errs

    datasets = sorted({r["dataset"] for r in rows})
    libs = {r["lib"] for r in rows}
    if len(libs) != 1:
        errs.append("the arms did not run the same binary: %s"
                    % ", ".join(sorted(libs)))

    for dataset in datasets:
        drows = [r for r in rows if r["dataset"] == dataset]
        bars = {r["bars"] for r in drows}
        if len(bars) != 1:
            errs.append("%s: the barcode differs between rows (%d distinct "
                        "values) -- the switch changed the RESULT, not just "
                        "the cost" % (dataset, len(bars)))
        # The VALUES, not just the number of the bars. A change that alters
        # the pairs while keeping their count would survive a count
        # comparison and still be a wrong answer.
        hashes = {r["barhash"] for r in drows if r["barhash"] != ""}
        if len(hashes) > 1:
            errs.append("%s: the barcode VALUES differ between rows (%d "
                        "distinct hashes) -- the switch changed the RESULT"
                        % (dataset, len(hashes)))
        # Same work, cheaper: the reduction rounds per dimension must agree.
        for key in [c for c in rows[0] if c.startswith("dim") and c.endswith("_rounds")]:
            vals = {r[key] for r in drows if r[key] != ""}
            if len(vals) > 1:
                errs.append("%s: %s differs between rows (%s) -- the arms did "
                            "not do the same work" % (dataset, key, ", ".join(sorted(vals))))

        for arm in ARMS:
            arows = [r for r in drows if r["arm"] == arm]
            if not arows:
                errs.append("%s: the arm %s is missing" % (dataset, arm))
        if any(e.startswith(dataset + ":") for e in errs):
            continue

        on = _median([float(r["wall_s"]) for r in drows
                      if r["arm"] == "early_on" and r["wall_s"]])
        off = _median([float(r["wall_s"]) for r in drows
                       if r["arm"] == "early_off" and r["wall_s"]])
        if on is not None and off is not None and off > 0:
            if on > off * REGRESSION_MARGIN:
                errs.append("%s: the early-on arm is SLOWER (%.3f s vs %.3f s, "
                            "%.3fx) -- the change is a regression"
                            % (dataset, on, off, on / off))
            if verbose:
                print("  %-5s early_on %.3f s  early_off %.3f s  -> %.3fx"
                      % (dataset, on, off, off / on))
    return errs


def mutate(src, dst, kind):
    """A corrupted copy of the table, for the negative self-test."""
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    if kind == "barcode_differs":
        rows[0]["bars"] = '{"0": 1}'
    elif kind == "barvalues_differ":
        rows[0]["barhash"] = "0" * 64
    elif kind == "rounds_differ":
        for r in rows:
            if r["dim1_rounds"] != "":
                r["dim1_rounds"] = str(int(r["dim1_rounds"]) + 1)
                break
    elif kind == "different_binary":
        rows[0]["lib"] = "/somewhere/else/canns_lib.so"
    elif kind == "slower":
        # Make the early-on arm look slower than the early-off arm.
        for r in rows:
            if r["arm"] == "early_on":
                r["wall_s"] = "%.3f" % (float(r["wall_s"]) * 1.5)
    elif kind == "noisy_host":
        for r in rows:
            r["foreign_cpu"] = "1.500"
    elif kind == "drop_a_row":
        rows = rows[1:]
    elif kind == "failed_run":
        rows[0]["verdict"] = "RC=-9"
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ("barcode_differs", "barvalues_differ", "rounds_differ", "different_binary",
             "slower", "noisy_host", "drop_a_row", "failed_run")


def main() -> int:
    ap = argparse.ArgumentParser()
    quiet_gate.add_arguments(ap)
    ap.add_argument("csv", nargs="?", default="cofacet_early_exit_ab.csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    errs = check(args.csv, max_foreign_cpu=args.max_foreign_cpu)
    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    datasets = sorted({r["dataset"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    print("datasets       : %s" % ", ".join(datasets))
    print("arms           : %s" % ", ".join(arms))
    print("rows in table  : %d" % len(rows))
    if errs:
        print("\nFAIL (%d problem(s)):" % len(errs))
        for e in errs[:40]:
            print("  - %s" % e)
        return 1
    print("\nPASS: the two arms ran the same binary on the same work, produced "
          "the same barcode, and the early-exit arm is not slower")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="eearly_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, verbose=False)
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
