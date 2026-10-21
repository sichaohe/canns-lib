#!/usr/bin/env python
"""Checker for the same-batch gap accounting.

THE RAW CSV IS THE DATA SOURCE. This script computes every number that
appears in the deliverable markdown, and it computes the denominator itself
from the EXPECTED grid instead of trusting what the runner reported.

WHAT IS CHECKED
  1. COMPLETENESS. Every (arm, rep) cell of the EXPECTED grid is present,
     with every stage row. A missing cell is a FAIL.
  2. CLOSURE. For every run, the stage breakdown must sum to that run's
     own library total, within 5% of it. A breakdown that does not close
     cannot be used to say "the gap is in stage X" -- so this is a hard
     FAIL, not a caveat. (This rule is the one that has been violated three
     times in this project; it is checked mechanically now.)
  3. SANITY. The process wall must be >= the library total for every run,
     and the arm medians must be within a sane factor of each other.
  4. THE GAP. The algorithmic gap (library totals) and the end-to-end gap
     (process walls) are computed and printed, with the difference between
     them attributed to per-language process startup rather than to the
     algorithm.

The `--self-test` mode tampers with a copy of the raw CSV and asserts the
checker REJECTS each tamper. A checker that only ever prints PASS proves
nothing about itself.

USAGE
    python check_gap.py <raw.csv> [--reps 5] [--self-test] [--summary out.csv]
"""

import argparse
import csv
import os
import statistics
import sys
import tempfile

ARMS = ["ours_gpu", "ours_gpu_phase", "ripserpp_build3", "ripserpp_build2"]
STAGES = [
    "setup", "edges", "dim0",
    "d1_scan", "d1_table", "d1_reduce", "d1_other",
    "d2_scan", "d2_table", "d2_reduce", "d2_other",
    "assemble", "other", "unattributed",
    "library_total", "process_wall", "peak_rss_kb",
]
# The stage rows that make up the breakdown (i.e. excluding the totals).
BREAKDOWN = [s for s in STAGES if s not in ("library_total", "process_wall",
                                            "peak_rss_kb")]
CLOSURE_TOL = 0.05


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cell_map(rows):
    cells = {}
    for r in rows:
        c = cells.setdefault((r["arm"], int(r["rep"])), {})
        c[r["stage"]] = float(r["seconds"])
        # has_breakdown: 0 means this build has no instrumentation, so its
        # stage rows are placeholders (everything in `unattributed`) and only
        # the total is real. Mixing such a cell with an instrumented one
        # would average a measurement with a non-measurement.
        c["_has_breakdown"] = int(r.get("has_breakdown", 1))
    return cells


def check(rows, reps, verbose=True):
    fails = []
    if not rows:
        return ["the CSV is empty"]
    cells = cell_map(rows)
    exp = [(a, r) for a in ARMS for r in range(1, reps + 1)]
    for key in exp:
        if key not in cells:
            fails.append("missing cell %s" % (key,))
    for key in sorted(set(cells) - set(exp)):
        fails.append("unexpected cell %s (not in the EXPECTED grid)" % (key,))
    for key, c in sorted(cells.items()):
        for s in STAGES:
            if s not in c:
                fails.append("cell %s is missing the %s row" % (key, s))
        if any(s not in c for s in STAGES):
            continue
        # --- closure: the breakdown must add up to the library total ------
        # Only for cells that HAVE a breakdown. A cell without one carries
        # no stage information at all, and pretending otherwise is how a
        # "the gap is in stage X" claim gets made on numbers that were
        # never measured.
        if c["_has_breakdown"]:
            ssum = sum(c[s] for s in BREAKDOWN)
            lt = c["library_total"]
            if lt <= 0:
                fails.append("cell %s has a non-positive library total" % (key,))
            elif abs(ssum - lt) / lt > CLOSURE_TOL:
                fails.append(
                    "cell %s does not close: stages sum to %.3f but the library "
                    "total is %.3f (%.1f%% off, tolerance %d%%)"
                    % (key, ssum, lt, 100.0 * abs(ssum - lt) / lt,
                       int(CLOSURE_TOL * 100))
                )
        else:
            # No breakdown: every stage must be zero except `unattributed`,
            # and `unattributed` must equal the total.
            for s in BREAKDOWN:
                if s == "unattributed":
                    continue
                if abs(c[s]) > 1e-9:
                    fails.append(
                        "cell %s claims to have no breakdown but reports a "
                        "non-zero %s" % (key, s)
                    )
            if abs(c["unattributed"] - c["library_total"]) > 1e-6:
                fails.append(
                    "cell %s has no breakdown and its unattributed time "
                    "(%.3f) does not equal its total (%.3f)"
                    % (key, c["unattributed"], c["library_total"])
                )
        # --- sanity -------------------------------------------------------
        if c["process_wall"] < c["library_total"] - 1e-6:
            fails.append(
                "cell %s: process wall %.3f is below the library total %.3f"
                % (key, c["process_wall"], c["library_total"])
            )
        if c["peak_rss_kb"] <= 0:
            fails.append("cell %s has no peak RSS" % (key,))

    if fails:
        return fails

    # --- the numbers ------------------------------------------------------
    def med(arm, stage):
        return statistics.median(
            [c[stage] for (a, _), c in cells.items() if a == arm]
        )

    def spread(arm, stage):
        v = [c[stage] for (a, _), c in cells.items() if a == arm]
        return max(v) - min(v)

    ours_lt = med("ours_gpu", "library_total")
    rp3_lt = med("ripserpp_build3", "library_total")
    rp2_lt = med("ripserpp_build2", "library_total")
    ours_w = med("ours_gpu", "process_wall")
    rp3_w = med("ripserpp_build3", "process_wall")
    rp2_w = med("ripserpp_build2", "process_wall")
    instr = rp3_w - rp2_w          # what the PROFILING instrumentation costs
    alg_gap = ours_lt - rp3_lt     # both library-internal, same batch
    e2e_gap = ours_w - rp2_w       # what a user waits for
    startup_ours = ours_w - ours_lt
    startup_rp = rp3_w - rp3_lt

    # Sanity: the two ripser++ builds must be in the same ballpark.
    if not (0.7 < rp2_lt / rp3_lt < 1.4):
        fails.append(
            "the two ripser++ builds disagree on the total "
            "(%.3f vs %.3f) -- the stage split may not be comparable"
            % (rp2_lt, rp3_lt)
        )

    if verbose:
        print("EXPECTED_CELLS %d (arms=%d x reps=%d)" % (len(exp), len(ARMS), reps))
        print("ROWS %d" % len(rows))
        print()
        print("=== totals (medians over %d reps) ===" % reps)
        for arm, label in (("ours_gpu", "canns-lib GPU"),
                           ("ripserpp_build3", "ripser++ build3 (PROFILING)"),
                           ("ripserpp_build2", "ripser++ build2 (baseline)")):
            print("  %-30s library=%8.3f  process=%8.3f  rss=%6.2f GB"
                  % (label, med(arm, "library_total"), med(arm, "process_wall"),
                     med(arm, "peak_rss_kb") / 1048576.0))
        print()
        print("  ALGORITHMIC gap (library totals, both instrumented): %+.3f s"
              % alg_gap)
        print("     less the PROFILING overhead (build3 wall - build2 wall):"
              " %.3f s  =>  %+.3f s against an uninstrumented ripser++"
              % (instr, alg_gap - instr))
        print("  END-TO-END  gap (process walls, build2 baseline):     %+.3f s"
              % e2e_gap)
        print("     process startup tax: ours %.3f s, ripser++ %.3f s"
              % (startup_ours, startup_rp))
        print()
        print("=== stage gap (library totals, medians) ===")
        print("  %-14s %9s %9s %9s" % ("stage", "ours", "ripser++", "delta"))
        for s in BREAKDOWN:
            o = med("ours_gpu", s)
            r = med("ripserpp_build3", s)
            mark = "  <<<" if abs(o - r) >= 0.5 else ""
            print("  %-14s %9.3f %9.3f %+9.3f%s"
                  % (s, o, r, o - r, mark))
        print()
        for arm in ARMS:
            print("  %-20s library_total spread=%.3f  process_wall spread=%.3f"
                  % (arm, spread(arm, "library_total"),
                     spread(arm, "process_wall")))

        # The fine-grained split of the GPU apparent pre-pass, from the
        # instrumented build. It is what makes the comparison with
        # ripser++'s PROFILING lines meaningful: the two implementations cut
        # their timers at different places.
        fine_stages = sorted({s for (a, _), c in cells.items()
                              for s in c if s.startswith("fine_")})
        if fine_stages:
            print()
            print("=== our GPU apparent pre-pass, split (instrumented arm) ===")
            print("  %-26s %9s" % ("phase", "seconds"))
            for s in fine_stages:
                print("  %-26s %9.3f" % (s.replace("fine_", ""),
                                         med("ours_gpu_phase", s)))
    return fails


def write_summary(path, rows, reps):
    cells = cell_map(rows)

    def med(arm, stage):
        return statistics.median(
            [c[stage] for (a, _), c in cells.items() if a == arm]
        )

    out = []
    ours_lt = med("ours_gpu", "library_total")
    rp3_lt = med("ripserpp_build3", "library_total")
    rp2_lt = med("ripserpp_build2", "library_total")
    ours_w = med("ours_gpu", "process_wall")
    rp3_w = med("ripserpp_build3", "process_wall")
    rp2_w = med("ripserpp_build2", "process_wall")
    out.append(dict(kind="gap", stage="algorithmic_gap",
                    ours=round(ours_lt, 4), rp=round(rp3_lt, 4),
                    rp_baseline="",
                    delta=round(ours_lt - rp3_lt, 4),
                    note="library totals, both instrumented, same batch; this is the gap to attack"))
    out.append(dict(kind="gap", stage="profiling_overhead",
                    ours="", rp=round(rp3_w, 4), rp_baseline=round(rp2_w, 4),
                    delta=round(rp3_w - rp2_w, 4),
                    note="what ripser++'s PROFILING build costs relative to the baseline build"))
    out.append(dict(kind="gap", stage="end_to_end_gap",
                    ours=round(ours_w, 4), rp="", rp_baseline=round(rp2_w, 4),
                    delta=round(ours_w - rp2_w, 4),
                    note="process walls; includes our Python/numpy startup"))
    out.append(dict(kind="gap", stage="startup_tax_ours",
                    ours=round(ours_w - ours_lt, 4), rp="", rp_baseline="",
                    delta="", note="our process wall minus our library total"))
    out.append(dict(kind="gap", stage="startup_tax_ripserpp",
                    ours="", rp=round(rp3_w - rp3_lt, 4), rp_baseline="",
                    delta="", note="ripser++'s process wall minus its library total"))
    for s in BREAKDOWN:
        out.append(dict(kind="stage", stage=s,
                        ours=round(med("ours_gpu", s), 4),
                        rp=round(med("ripserpp_build3", s), 4),
                        rp_baseline="",
                        delta=round(med("ours_gpu", s) - med("ripserpp_build3", s), 4),
                        note="stage split from the PROFILING build"))
    for arm in ARMS:
        out.append(dict(kind="arm", stage=arm,
                        ours="", rp="", rp_baseline="",
                        delta="",
                        note="library_total median %.3f, process_wall median %.3f, peak RSS %.2f GB"
                             % (med(arm, "library_total"), med(arm, "process_wall"),
                                med(arm, "peak_rss_kb") / 1048576.0)))
    # The fine-grained split, for the stages that exist.
    fine = sorted({s for (a, _), c in cells.items()
                    for s in c if s.startswith("fine_")})
    for s in fine:
        out.append(dict(kind="fine", stage=s.replace("fine_", ""),
                        ours=round(med("ours_gpu_phase", s), 4),
                        rp="", rp_baseline="", delta="",
                        note="our GPU apparent pre-pass, split by the GPUPHASE timers"))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "stage", "ours", "rp",
                                           "rp_baseline", "delta", "note"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    return path


def self_test(path, reps):
    base = load(path)
    tampers = []

    def t(name, fn):
        tampers.append((name, fn))

    def drop_cell(rows):
        return [r for r in rows if not (r["arm"] == "ripserpp_build3"
                                        and r["rep"] == "2")]

    def break_closure(rows):
        r = [dict(x) for x in rows]
        for x in r:
            if x["arm"] == "ours_gpu" and x["stage"] == "d2_reduce":
                x["seconds"] = str(float(x["seconds"]) * 3.0)
                break
        return r

    def break_wall(rows):
        r = [dict(x) for x in rows]
        for x in r:
            if x["arm"] == "ours_gpu" and x["stage"] == "process_wall":
                x["seconds"] = "1.0"
                break
        return r

    def break_rss(rows):
        r = [dict(x) for x in rows]
        for x in r:
            if x["stage"] == "peak_rss_kb":
                x["seconds"] = "0"
                break
        return r

    def drop_stage(rows):
        r = [dict(x) for x in rows]
        for i, x in enumerate(r):
            if x["stage"] == "d2_table":
                del r[i]
                break
        return r

    t("a cell is missing", drop_cell)
    t("a stage does not close to the library total", break_closure)
    t("a process wall is below the library total", break_wall)
    t("a peak RSS is missing", break_rss)
    t("a stage row is missing", drop_stage)

    ok = True
    for name, fn in tampers:
        rows = fn(base)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(base[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
            tmp = f.name
        try:
            fails = check(load(tmp), reps, verbose=False)
        finally:
            os.unlink(tmp)
        if not fails:
            print("  SELF-TEST FAILED: the checker accepted %r" % name)
            ok = False
        else:
            print("  rejected as it must be: %-44s -> %s" % (name, fails[0]))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--summary", default=None,
                    help="write the computed summary table to this CSV")
    args = ap.parse_args()

    if args.self_test:
        print("=== negative self-test ===")
        good = self_test(args.csv, args.reps)
        print("SELF-TEST", "PASS" if good else "FAIL")
        return 0 if good else 1

    rows = load(args.csv)
    fails = check(rows, args.reps)
    if args.summary:
        write_summary(args.summary, rows, args.reps)
        print("wrote %s" % args.summary)
    if fails:
        print("FAIL")
        for f in fails:
            print("  ! " + f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
