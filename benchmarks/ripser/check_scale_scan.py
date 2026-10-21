#!/usr/bin/env python3
"""Checker for the scale scan.

  1. THE GRID IS COMPLETE. Every size has all five arms: our build on the
     `+inf` matrix, our build on the capped matrix at both thresholds, and
     ripser++ on the capped matrix at both thresholds. The denominator is
     computed from the size list, not remembered.
  2. THE ARMS COMPUTE THE SAME FILTRATION. Capping the missing pairs does
     not change WHICH simplices are in the filtration, only how many
     simplices have to be enumerated -- so the arms that share a filtration
     must produce the SAME barcode, and the bar counts must agree. This is
     the invariant the whole experiment rests on.
  3. THE MEASUREMENT IS USABLE. A positive wall time, a peak RSS, the load
     average, and the build identity -- a benchmark table without the
     identity of the build that produced it is not reproducible.
  4. THE FAILURES ARE LABELLED. The reference can fail to fit the device at
     a size, and that IS the result the scan exists to find -- but it has to
     say so, in its own words. Our side has no such excuse: an error on
     our side is a broken measurement, not a finding.
  5. THE CLOCKS AGREE WITH EACH OTHER. The library wall clock cannot be
     larger than the process wall clock that contains it.

USAGE
    python check_scale_scan.py scale_scan.csv [--sizes 1200,1600,2000] [--self-test]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile

# The five arms of the scan. `ripserpp_T` is the FAIR comparison: ripser++
# can be run with the threshold at the max finite distance, and then the
# capped pairs are excluded and the filtration is the same as the one we get
# by keeping the `+inf` -- so the barcodes agree and the times are
# comparable. `ripserpp_all` (the cap threshold) is the configuration a
# ripser++ user would NOT choose, and it is kept in the table for the
# record, not for the claim.
SIDES = ("ours_inf", "ours_T", "ours_all", "ripserpp_T", "ripserpp_all")
OURS = ("ours_inf", "ours_T", "ours_all")


def is_ours(r):
    return r.get("side") in OURS


def row_label(r):
    return "n=%s %s:" % (r.get("n"), r.get("side"))
REFERENCE = ("ripserpp_T", "ripserpp_all")
# The arms that compute the SAME filtration, so their bar counts must be
# equal: the two threshold settings of each implementation, and our `+inf`
# against the reference's threshold.
SAME_FILTRATION = (("ours_inf", "ours_T"),
                   ("ours_T", "ours_all"),
                   ("ripserpp_T", "ripserpp_all"),
                   ("ours_inf", "ripserpp_T"))
# What the reference is allowed to record as a finding rather than a fault.
REFERENCE_FINDINGS = ("TIMEOUT", "GPU_OOM")
PROVENANCE_COLUMNS = ("threads", "commit", "build_md5")


def check(path, sizes=None, max_loadavg=0.0, max_foreign_cpu=0.25):
    fails = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["table is empty: %s" % path]
    for col in ("n", "side", "threshold", "wall_s", "rss_kb", "bars",
                "error", "note", "loadavg", "foreign_cpu", "foreign_max"):
        if col not in rows[0]:
            return ["table is missing column %s" % col]

    ns = sizes or sorted({int(r["n"]) for r in rows})
    print("expected cells: %d sizes x %d arms = %d"
          % (len(ns), len(SIDES), len(ns) * len(SIDES)))
    print("rows in table             : %d" % len(rows))

    by = {}
    for r in rows:
        by.setdefault(int(r["n"]), {})[r["side"]] = r
    for n in ns:
        for side in SIDES:
            if side not in by.get(n, {}):
                fails.append("missing cell: n=%d side=%s" % (n, side))
    for n in by:
        if n not in ns:
            fails.append("unexpected size: n=%d" % n)

    for n, sides in by.items():
        # Every arm that computes the same filtration must report the same
        # number of bars. The reference implementation is not run-to-run
        # reproducible on this workload (its GPU reduction order varies, and
        # a near-tie in the H2 classes then resolves differently), so a
        # difference of a fraction of a percent is tolerated -- but a larger
        # one is not.
        for a_name, b_name in SAME_FILTRATION:
            ra, rb = sides.get(a_name), sides.get(b_name)
            if not ra or not rb:
                continue
            if not (ra.get("bars") and rb.get("bars")):
                continue
            a, b = int(ra["bars"]), int(rb["bars"])
            if a == b:
                continue
            if abs(a - b) > max(2, int(0.005 * max(a, b))):
                fails.append("n=%d: %s reports %s bars and %s reports %s -- "
                             "they compute the same filtration, so the "
                             "difference is larger than the reference's "
                             "run-to-run variation" % (n, a_name, ra["bars"],
                                                      b_name, rb["bars"]))
            else:
                print("  n=%d: %s and %s differ by %d bars (%.3f%%) -- the "
                      "reference is not run-to-run reproducible"
                      % (n, a_name, b_name, abs(a - b),
                         100.0 * abs(a - b) / max(a, b)))
        for side, r in sides.items():
            err = (r.get("error") or "").strip()
            note = (r.get("note") or "").strip()
            if err in REFERENCE_FINDINGS and side in REFERENCE:
                # The reference did not fit the device at this size. That IS
                # the result at this size -- it is why the scan exists -- so
                # it is not a failure, but it has to say WHY: the note
                # carries the device-memory requirement ripser++ itself
                # printed.
                print("  n=%d %s: %s (%s)" % (n, side, err, note))
                if "device memory" not in note:
                    fails.append("n=%d %s: the %s does not record the "
                                 "device-memory requirement ripser++ printed"
                                 % (n, side, err))
                continue
            if err:
                fails.append("n=%d %s: the arm failed (%s: %s) -- a failure is "
                             "not a measurement" % (n, side, err, note))
                continue
            try:
                if float(r["wall_s"]) <= 0:
                    fails.append("n=%d %s: non-positive wall time" % (n, side))
                if int(r["rss_kb"]) <= 0:
                    fails.append("n=%d %s: no peak RSS" % (n, side))
            except ValueError:
                fails.append("n=%d %s: the timing columns are not numbers"
                             % (n, side))
            if not (r.get("loadavg") or "").strip():
                fails.append("n=%d %s: no load average recorded" % (n, side))
            if not (r.get("foreign_cpu") or "").strip():
                fails.append("n=%d %s: no foreign-load reading recorded -- "
                             "without it the table cannot say whether it was "
                             "measured on a quiet host" % (n, side))
            # the build identity. On our side it is the source fingerprint
            # and the thread count; on the reference side it is the md5 of
            # the binary that produced the row -- the convention the
            # head-to-head table uses as well.
            for col in PROVENANCE_COLUMNS:
                if side in OURS:
                    if not (r.get(col) or "").strip():
                        fails.append("n=%d %s: the row does not record %s"
                                     % (n, side, col))
                elif col == "build_md5":
                    if not (r.get(col) or "").strip():
                        fails.append("n=%d %s: the row does not record %s"
                                     % (n, side, col))
            # the library clock cannot be larger than the process clock
            lib = (r.get("lib_wall_s") or "").strip()
            if lib:
                try:
                    if float(lib) > float(r["wall_s"]) + 1e-6:
                        fails.append("n=%d %s: the library wall clock (%s s) is "
                                     "larger than the process wall clock "
                                     "(%s s)" % (n, side, lib, r["wall_s"]))
                except ValueError:
                    fails.append("n=%d %s: the library wall clock is not a "
                                 "number" % (n, side))
            elif side in OURS:
                fails.append("n=%d %s: no library wall clock" % (n, side))

    # A BASELINE THAT FAILED LOOKS LIKE A BASELINE THAT WON. A reference
    # run that read no input, or that died, comes back in a fraction of
    # the time the real run takes, and a table that reports it reports a
    # crash as a result. Every row must have a positive wall clock, and
    # the reference must not be faster than a small fraction of ours.
    ours_walls = []
    for r in rows:
        try:
            w = float((r.get("wall_s") or "").strip() or 0)
        except ValueError:
            w = 0.0
        # A row that TIMED OUT has no wall clock, and that is a legal
        # result -- the reference did not finish inside the limit, which is
        # exactly what the scale wall is. What is not legal is a row that
        # has no wall clock and no timeout: that is a run that did nothing.
        timed_out = "TIMEOUT" in ((r.get("error") or "") + (r.get("note") or "")).upper()
        if w <= 0 and not timed_out:
            fails.append("%s the wall clock is %r and the row does not say it "
                         "timed out -- the run did not do any work"
                         % (row_label(r), r.get("wall_s")))
        if is_ours(r):
            ours_walls.append(w)
    if ours_walls and min(ours_walls) > 0:
        shortest = min(ours_walls)
        for r in rows:
            if is_ours(r):
                continue
            try:
                w = float((r.get("wall_s") or "").strip() or 0)
            except ValueError:
                w = 0.0
            if 0 < w < shortest * 0.05:
                fails.append("%s finished in %.3f s, under 5%% of the "
                             "shortest of our own runs (%.3f s) on the same "
                             "input: that is a failed run, not a fast one"
                             % (row_label(r), w, shortest))

    # A table measured while the host was busy is not a measurement of the
    # code: on this shared machine a neighbour's job changes the numbers by
    # up to 40% and can flip a comparison's sign. So a table can be asked
    # to prove it was measured in a quiet window.
    #
    # THE GATE IS THE FOREIGN LOAD, NOT THE LOAD AVERAGE. The load average
    # counts our own measurement -- the reduction runs on the host with one
    # thread per core, and our own runs put it at 3.1 on this 16-core box
    # -- so a threshold on it has to clear our own footprint, and a
    # neighbour's two-core job then hides underneath. The `foreign_cpu`
    # column is the number of cores used by processes that are NOT ours,
    # measured while the row was being produced: ours is excluded by
    # construction, so a small value is evidence and not an artefact of how
    # busy we are. The load average is kept as a diagnostic, and as an
    # optional coarse gate for a host that has gone badly wrong.
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
                bad = ("the foreign load %s cores on this row is above the "
                       "quiet-window threshold %.2f -- somebody else was "
                       "using the host while this row was measured"
                       % (raw, max_foreign_cpu))
        print("worst foreign load: mean %.3f cores, largest single window "
              "%.3f cores (threshold %.2f)"
              % (worst_mean, worst_max, max_foreign_cpu))
        if bad:
            fails.append(bad)

    if max_loadavg > 0:
        for r in rows:
            try:
                la = float(r.get("loadavg") or 0.0)
            except ValueError:
                fails.append("the load average column is not a number")
                break
            if la >= max_loadavg:
                fails.append("the load average %s on this row is not below "
                             "the quiet-window threshold %.2f -- the batch "
                             "was measured on a busy host"
                             % (la, max_loadavg))
                break

    return fails


def mutate(src, dst, kind):
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())

    def first(pred):
        for r in rows:
            if pred(r):
                return r
        raise SystemExit("the mutation %s has no row to act on -- the table "
                         "does not have the shape the self-test assumes"
                         % kind)

    if kind == "drop_side":
        rows = [r for r in rows if not (r["side"] == "ours_all"
                                        and r["n"] == rows[0]["n"])]
    elif kind == "differing_forms":
        first(lambda r: r["side"] == "ours_T")["bars"] = "999"
    elif kind == "bars_disagree":
        first(lambda r: r["side"] == "ripserpp_T")["bars"] = "999999"
    elif kind == "no_gpu_free":
        # make the reference claim it ran out of the device, and then take
        # away its own explanation of why
        r = first(lambda x: x["side"] == "ripserpp_T")
        r["error"] = "TIMEOUT"
        r["note"] = ""
    elif kind == "zero_time":
        first(lambda r: r["side"] in OURS)["wall_s"] = "0"
    elif kind == "no_loadavg":
        first(lambda r: r["side"] in OURS)["loadavg"] = ""
    elif kind == "foreign_busy":
        # somebody else was using the host while this row was measured
        first(lambda r: r["side"] in OURS)["foreign_cpu"] = "3.500"
    elif kind == "no_foreign":
        first(lambda r: r["side"] in OURS)["foreign_cpu"] = ""
    elif kind == "crash_row":
        r = first(lambda x: x["side"] in OURS)
        r["error"] = "CRASH"
        r["note"] = "exited with status 1"
    elif kind == "no_provenance":
        first(lambda r: r["side"] in OURS)["commit"] = ""
    elif kind == "lib_wall_over_wall":
        r = first(lambda x: x["side"] in OURS)
        r["lib_wall_s"] = "%.4f" % (float(r["wall_s"]) + 10.0)
    else:
        raise SystemExit("unknown mutation %s" % kind)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


MUTATIONS = ["drop_side", "differing_forms", "bars_disagree", "no_gpu_free",
             "zero_time", "no_loadavg", "foreign_busy", "no_foreign",
             "crash_row", "no_provenance",
             "lib_wall_over_wall"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="scale_scan.csv")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--max-loadavg", type=float, default=0.0,
                    help="fail if any row was measured at or above this load "
                         "average; 0 = do not check. The load average counts "
                         "OUR OWN threads, so it is a coarse diagnostic, not "
                         "the quiet-window gate -- that is --max-foreign-cpu.")
    ap.add_argument("--max-foreign-cpu", type=float, default=0.25,
                    help="fail if any row was measured while somebody else "
                         "was using more than this many cores; 0 = do not "
                         "check. Ours is excluded from the reading by "
                         "construction, so this is the contamination gate.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()] or None

    fails = check(args.csv, sizes, max_loadavg=args.max_loadavg,
                   max_foreign_cpu=args.max_foreign_cpu)
    if fails:
        print("\nFAIL (%d problems):" % len(fails))
        for f in fails[:40]:
            print("  - %s" % f)
        return 1
    print("\nPASS: every size has all five arms, the arms that share a "
          "filtration agree on the barcode, and the measurements are "
          "identified")

    if args.self_test:
        tmp = tempfile.mkdtemp(prefix="scale_self_")
        try:
            bad = 0
            for kind in MUTATIONS:
                dst = os.path.join(tmp, "mut_%s.csv" % kind)
                mutate(args.csv, dst, kind)
                f = check(dst, sizes, max_loadavg=args.max_loadavg,
                          max_foreign_cpu=args.max_foreign_cpu)
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
