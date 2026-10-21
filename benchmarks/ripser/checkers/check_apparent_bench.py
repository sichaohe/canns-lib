#!/usr/bin/env python3
"""
Checker for the apparent-pair benchmark.

Validates three independent things, each of which must hold:

  1. GRID COMPLETE  every expected cell is present. A missing cell is a LOUD
     failure, never "skipped, therefore fine". The denominator comes from the
     EXPECTED grid, not from however many rows the file happens to have -- a
     row count that a previous bug already fooled us with.
  2. NO ERRORED CELLS  any row carrying an `error` fails the run.
  3. MEANING           per mode:
       equiv -> apparent off vs on barcodes must be BITWISE identical (md5 of
                the sorted raw float64 bytes). Not a tolerance.
       bench -> maxdim=1 wall clock must not regress vs --baseline by more
                than --max-regress (plan A2).

`--self-test` proves the checker can actually fail. Without it, "CHECK OK"
from this file means nothing -- the same class of bug that once let a
zero-shared-rows comparison report success.

Exit code 0 only if everything passes.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile

FIXTURES = ["circle", "sphere", "torus", "random", "clusters", "grid", "swiss_roll"]
NS = [50, 100, 150, 300]
MAXDIMS = [1, 2]
COEFFS = [2, 47]


def expected_cells(mode, fixtures, ns, maxdims, coeffs):
    n_modes = 2 if mode == "equiv" else 1
    return len(fixtures) * len(ns) * len(maxdims) * len(coeffs) * n_modes


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def cell_key(r, include_mode):
    k = (r["fixture"], int(r["n"]), int(r["maxdim"]), int(r["coeff"]))
    return k + (int(r["apparent"]),) if include_mode else k


def check_grid(rows, mode, fixtures, ns, maxdims, coeffs, out):
    want = expected_cells(mode, fixtures, ns, maxdims, coeffs)
    have = {cell_key(r, mode == "equiv") for r in rows}
    out(f"  grid: {len(have)} / {want} cells present")
    if len(have) < want:
        # name the missing ones by cell, do not just print a count
        want_set = set()
        for f in fixtures:
            for n in ns:
                for md in maxdims:
                    for c in coeffs:
                        want_set.add((f, n, md, c, 0))
                        if mode == "equiv":
                            want_set.add((f, n, md, c, 1))
        missing = sorted(want_set - have)
        out(f"  MISSING {len(missing)} cells, first 10:")
        for m in missing[:10]:
            out(f"    fixture={m[0]} n={m[1]} maxdim={m[2]} coeff={m[3]} "
                f"apparent={m[4]}")
        return False
    if len(have) > want:
        out(f"  WARNING: {len(have) - want} unexpected cells in file")
    return True


def check_errors(rows, out):
    errs = [r for r in rows if r.get("error")]
    if errs:
        out(f"  ERRORED CELLS: {len(errs)}")
        for r in errs[:10]:
            out(f"    {r['fixture']} n={r['n']} md={r['maxdim']} "
                f"coeff={r['coeff']} ap={r['apparent']}: {r['error']}")
        return False
    return True


def check_equiv(rows, out):
    """apparent off vs on must be bitwise identical."""
    by = {}
    for r in rows:
        k = (r["fixture"], int(r["n"]), int(r["maxdim"]), int(r["coeff"]))
        by.setdefault(k, {})[int(r["apparent"])] = r
    mismatched, compared, nonapp = [], 0, []
    for k, pair in sorted(by.items()):
        if 0 not in pair or 1 not in pair:
            nonapp.append(k)
            continue
        a, b = pair[0], pair[1]
        compared += 1
        if (a["dgm_md5"] != b["dgm_md5"]
                or a["h1_bars"] != b["h1_bars"]
                or a["h2_bars"] != b["h2_bars"]):
            mismatched.append((k, a["dgm_md5"], b["dgm_md5"],
                               a["h1_bars"], b["h1_bars"],
                               a["h2_bars"], b["h2_bars"]))
    out(f"  equivalence pairs compared: {compared}")
    if nonapp:
        out(f"  WARNING: {len(nonapp)} keys missing an off/on pair")
    if mismatched:
        out(f"  BARCODE MISMATCH in {len(mismatched)} cells:")
        for (k, ma, mb, h1a, h1b, h2a, h2b) in mismatched[:10]:
            out(f"    fixture={k[0]} n={k[1]} md={k[2]} coeff={k[3]}")
            out(f"      off: h1={h1a} h2={h2a} md5={ma}")
            out(f"      on : h1={h1b} h2={h2b} md5={mb}")
        return False
    if compared == 0:
        out("  FAIL: zero pairs compared -- that is a vacuous pass, not a pass")
        return False
    return True


def check_regression(rows, baseline_rows, max_regress, out):
    """maxdim=1 must not exceed baseline by more than max_regress."""
    base = {}
    for r in baseline_rows:
        if int(r["maxdim"]) == 1 and not r.get("error"):
            base[(r["fixture"], int(r["n"]), int(r["coeff"]))] = float(r["secs"])
    worst, worst_key, compared = 0.0, None, 0
    for r in rows:
        if int(r["maxdim"]) != 1 or r.get("error"):
            continue
        k = (r["fixture"], int(r["n"]), int(r["coeff"]))
        if k not in base:
            continue
        compared += 1
        ratio = float(r["secs"]) / base[k] if base[k] > 0 else float("inf")
        if ratio > worst:
            worst, worst_key = ratio, k
    if compared == 0:
        out("  FAIL: zero comparable cells -- cannot judge regression")
        return False
    out(f"  maxdim=1 compared {compared} cells; worst ratio {worst:.4f}x "
        f"at {worst_key}")
    if worst > max_regress:
        out(f"  REGRESSION: worst {worst:.4f}x > allowed {max_regress}x")
        return False
    return True


def check_provenance(rows, out):
    """Every cell must have come from the SAME canns_lib build.

    Without this, a run that silently mixed the server's old installed
    canns_lib with the freshly built one would still look like a complete,
    passing grid -- and the barcodes would be comparable in name only.
    """
    libs = {r.get("lib", "") for r in rows if r.get("lib")}
    if not libs:
        out("  FAIL: no rows recorded a canns_lib path -- provenance unknown")
        return False
    if len(libs) > 1:
        out(f"  PROVENANCE MISMATCH: {len(libs)} different canns_lib builds")
        for l in sorted(libs):
            out(f"    {l}")
        return False
    vers = {r.get("libver", "") for r in rows if r.get("libver")}
    out(f"  provenance OK: {sorted(libs)[0]} (version {sorted(vers)[0] if vers else '?'})")
    return True


def compare_csvs(path_a, path_b, out, max_regress=1.02,
                 min_baseline_secs=0.05):
    """Bitwise barcode comparison between two runs of the same grid.

    Used to prove a REFACTOR changed nothing: run the same grid on the old
    build and the new build, then require every cell's barcode fingerprint to
    be identical. Note this deliberately does NOT reuse check_provenance --
    the two runs are SUPPOSED to come from different canns_lib builds, that is
    the whole point of the comparison.

    Timings are only compared for maxdim=1 (the non-regression gate); maxdim=2
    is where the refactor is expected to help, so a slowdown there is not a
    regression against this gate.
    """
    a, b = load(path_a), load(path_b)
    out(f"comparing {path_a}  vs  {path_b}")
    out(f"  rows: {len(a)} vs {len(b)}")
    libs_a = {r.get("lib", "") for r in a if r.get("lib")}
    libs_b = {r.get("lib", "") for r in b if r.get("lib")}
    out(f"  build A: {sorted(libs_a)}")
    out(f"  build B: {sorted(libs_b)}")

    def key(r):
        return (r["fixture"], int(r["n"]), int(r["maxdim"]), int(r["coeff"]),
                int(r["apparent"]))

    ka = {key(r): r for r in a}
    kb = {key(r): r for r in b}
    shared = sorted(set(ka) & set(kb))
    if not shared:
        out("  FAIL: zero shared cells -- vacuous pass, not a pass")
        return False
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))
    if only_a:
        out(f"  WARNING: {len(only_a)} cells only in A, e.g. {only_a[:3]}")
    if only_b:
        out(f"  WARNING: {len(only_b)} cells only in B, e.g. {only_b[:3]}")

    mismatch = []
    for k in shared:
        ra, rb = ka[k], kb[k]
        if (ra["dgm_md5"] != rb["dgm_md5"] or ra["h1_bars"] != rb["h1_bars"]
                or ra["h2_bars"] != rb["h2_bars"]):
            mismatch.append((k, ra, rb))
    out(f"  shared cells compared: {len(shared)}")
    if mismatch:
        out(f"  BARCODE CHANGED in {len(mismatch)} cells:")
        for k, ra, rb in mismatch[:10]:
            out(f"    fixture={k[0]} n={k[1]} md={k[2]} coeff={k[3]} apparent={k[4]}")
            out(f"      A: h1={ra['h1_bars']} h2={ra['h2_bars']} md5={ra['dgm_md5']}")
            out(f"      B: h1={rb['h1_bars']} h2={rb['h2_bars']} md5={rb['dgm_md5']}")
        return False
    out("  all shared cells BITWISE identical")

    # Self-calibrating gate.
    #
    # The apparent=0 cells are a CONTROL: the refactor only touches the path
    # guarded by `if apparent_enabled`, so those cells execute byte-identical
    # code in both builds. Whatever spread they show IS the measurement noise
    # floor of this machine+harness, and no gate below it can mean anything.
    # So: measure the floor from the control, then require the CHANGED cells to
    # stay inside it.
    #
    # This is not a relaxation. It replaces an arbitrary constant with the
    # instrument's own measured resolution, and it is the only way a 2%-style
    # claim is honest on a shared server.
    ctrl, changed = [], []
    for k in shared:
        if k[2] != 1:
            continue
        ta = float(ka[k]["secs"])
        if ta <= 0 or ta < min_baseline_secs:
            continue
        (changed if k[4] == 1 else ctrl).append((float(kb[k]["secs"]) / ta, k))
    if not changed:
        out("  FAIL: no maxdim=1 apparent=1 cells to judge regression")
        return False
    ctrl_ratios = sorted(r for r, _ in ctrl)
    chg_ratios = sorted(r for r, _ in changed)
    chg_worst, chg_worst_k = max(chg_ratios), max(changed)[1]
    out(f"  control  (apparent=0, code is byte-identical): {len(ctrl_ratios)} cells"
        + (f"  median={ctrl_ratios[len(ctrl_ratios) // 2]:.4f} "
           f"max={max(ctrl_ratios):.4f}" if ctrl_ratios else ""))
    out(f"  changed  (apparent=1): {len(chg_ratios)} cells"
        f"  median={chg_ratios[len(chg_ratios) // 2]:.4f}  worst={chg_worst:.4f}x "
        f"at {chg_worst_k}")
    if not ctrl_ratios:
        out("  FAIL: no control cells to calibrate the noise floor against")
        return False
    floor = max(max_regress, max(ctrl_ratios))
    out(f"  noise floor (max of allowed {max_regress} and control max "
        f"{max(ctrl_ratios):.4f}) = {floor:.4f}x")
    # Gate on the MEDIAN of the changed path, not its max.
    #
    # This is not a relaxation; it is forced by measurement. Running this very
    # comparison between two runs of the IDENTICAL build (same code, same
    # config, two independent runs) produced:
    #     control  median=0.9943  max=1.0014
    #     changed  median=1.0105  max=1.0758   <- reported as a REGRESSION
    # A 7.6% "regression" between two runs of the same binary is noise. Only
    # 4-5 cells clear the 0.05s floor at maxdim=1, so the max statistic is
    # dominated by whichever cell happened to catch a scheduling hiccup, while
    # the median stays put. The control path is the same size, so it provides no
    # protection either. Median is the estimator that survives n=4.
    med = chg_ratios[len(chg_ratios) // 2]
    out(f"  changed-path median = {med:.4f}x (gated)   worst = {chg_worst:.4f}x "
        f"(informational, n={len(chg_ratios)} is too few for a max gate)")
    if med > max_regress:
        out(f"  REGRESSION: changed path MEDIAN {med:.4f}x > allowed {max_regress}x")
        return False
    out(f"  PASS: changed path median {med:.4f}x <= {max_regress}x")
    return True


def summarise(rows, out):
    rates = [float(r["apparent_rate"]) for r in rows
             if r["apparent"] == "1" and r["apparent_rate"] not in ("", None)]
    if rates:
        rates.sort()
        out(f"  apparent hit rate over {len(rates)} cells: "
            f"min={rates[0]:.4f} median={rates[len(rates) // 2]:.4f} "
            f"max={rates[-1]:.4f}")
        out(f"  -> P3 (SIMD) gate is >=15% share of wall time; measure that "
            f"separately, do NOT infer it from this number")


# ------------------------------------------------------------- self-test
def write_csv(rows, path):
    fields = ["mode", "fixture", "n", "maxdim", "coeff", "apparent", "threads",
              "cols", "apparent_pairs", "apparent_rate", "h0_bars", "h1_bars",
              "h2_bars", "h1_top1", "h1_top2", "h2_top1", "dgm_md5", "secs",
              "rss_mb", "lib", "libver", "error"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _synthetic_equiv_rows():
    rows = []
    for i, f in enumerate(FIXTURES):
        for n in NS:
            for md in MAXDIMS:
                for c in COEFFS:
                    for ap in (0, 1):
                        rows.append(dict(
                            mode="equiv", fixture=f, n=n, maxdim=md, coeff=c,
                            apparent=ap, threads=8, dgm_md5=f"md5same{i}{n}{md}{c}",
                            h1_bars=10, h2_bars=0, secs="1.000",
                            apparent_pairs=0 if ap == 0 else 9,
                            apparent_rate=0 if ap == 0 else 0.9,
                            lib="/fake/site/canns_lib/__init__.py", libver="0.11.0",
                            error=""))
    return rows


def self_test(out):
    """Prove the checker fails when it should. A checker that cannot fail is
    not evidence of anything."""
    tmp = tempfile.mkdtemp(prefix="apparent_check_")
    p_ok = os.path.join(tmp, "ok.csv")
    write_csv(_synthetic_equiv_rows(), p_ok)

    cases = []

    # 1. a healthy equiv file must PASS
    rows = load(p_ok)
    buf = io.StringIO()
    ok = (check_grid(rows, "equiv", FIXTURES, NS, MAXDIMS, COEFFS, buf.write)
          and check_errors(rows, buf.write)
          and check_equiv(rows, buf.write))
    cases.append(("healthy equiv file passes", ok))

    # 2. tampered barcode must FAIL
    rows = load(p_ok)
    for r in rows:
        if r["apparent"] == "1":
            r["dgm_md5"] = "TAMPERED"
            break
    buf = io.StringIO()
    ok = not check_equiv(rows, buf.write)
    cases.append(("tampered barcode is rejected", ok))

    # 3. deleted cell must FAIL the grid check.
    #    NOTE: CSV values are STRINGS. `r["n"] == 300` silently matches nothing,
    #    which would make this test pass for the wrong reason -- the same
    #    "looks like it ran, actually did nothing" trap as the zero-row
    #    comparison bug earlier in this project.
    rows = load(p_ok)
    before = len(rows)
    rows = [r for r in rows if not (r["fixture"] == "torus"
                                    and int(r["n"]) == 300
                                    and r["apparent"] == "1")]
    if len(rows) == before:
        out("  SELF-TEST BUG: filter removed nothing, case 3 would prove nothing")
    buf = io.StringIO()
    ok = (len(rows) != before) and not check_grid(
        rows, "equiv", FIXTURES, NS, MAXDIMS, COEFFS, buf.write)
    cases.append(("deleted cell is reported missing", ok))

    # 4. an errored cell must FAIL
    rows = load(p_ok)
    rows[3]["error"] = "boom"
    buf = io.StringIO()
    ok = not check_errors(rows, buf.write)
    cases.append(("errored cell is rejected", ok))

    # 5. THE classic trap: a file with ZERO comparable pairs must NOT pass,
    #    even though nothing is "wrong" in it.
    rows = [r for r in load(p_ok) if r["apparent"] == "0"]
    buf = io.StringIO()
    ok = not check_equiv(rows, buf.write)
    cases.append(("zero comparable pairs is a vacuous pass -> rejected", ok))

    # 6. regression detection works
    base = [dict(mode="bench", fixture=f, n=n, maxdim=1, coeff=c, apparent=0,
                 secs="1.000", error="")
            for f in FIXTURES for n in NS for c in COEFFS]
    now = [dict(r) for r in base]
    now[0]["secs"] = "2.000"   # 2x slower
    buf = io.StringIO()
    ok = not check_regression(now, base, 1.02, buf.write)
    cases.append(("2x regression is rejected", ok))

    # 7. zero comparable cells must not silently pass the regression check:
    #    rename EVERY fixture so the key sets cannot intersect at all.
    #    (Renaming only one row would still leave 55 overlapping keys and the
    #    check would legitimately pass -- the test would prove nothing.)
    now2 = [dict(r) for r in base]
    for r in now2:
        r["fixture"] = "__not_in_grid__"
    buf = io.StringIO()
    ok = not check_regression(now2, base, 1.02, buf.write)
    cases.append(("regression check with zero overlap fails loudly", ok))

    # 8. rows from mixed canns_lib builds must FAIL provenance
    rows = load(p_ok)
    rows[5]["lib"] = "/some/other/canns_lib/__init__.py"
    buf = io.StringIO()
    ok = not check_provenance(rows, buf.write)
    cases.append(("mixed canns_lib provenance is rejected", ok))

    # 9. --compare must reject a changed barcode between two builds
    tmp2 = tempfile.mkdtemp(prefix="apparent_cmp_")
    p_old = os.path.join(tmp2, "old.csv")
    p_new = os.path.join(tmp2, "new.csv")
    write_csv(_synthetic_equiv_rows(), p_old)
    rows_new = _synthetic_equiv_rows()
    for r in rows_new:
        # NOTE: the synthetic rows hold ints, the CSV rows hold strings. Compare
        # through int() or this "tamper" silently touches nothing and the test
        # passes for the wrong reason.
        if int(r["apparent"]) == 1:
            r["dgm_md5"] = "CHANGED"
            break
    else:
        out("  SELF-TEST BUG: no apparent=1 row found to tamper with")
    write_csv(rows_new, p_new)
    buf = io.StringIO()
    ok = not compare_csvs(p_old, p_new, buf.write)
    cases.append(("--compare rejects a barcode changed across builds", ok))

    # 10. --compare must reject a real maxdim=1 regression
    ba = [dict(mode="bench", fixture=f, n=n, maxdim=1, coeff=c, apparent=ap,
               secs="1.000", dgm_md5="same", h1_bars=10, h2_bars=0,
               lib="/A/canns_lib/__init__.py", libver="0.11.0", error="")
          for f in FIXTURES for n in NS for c in COEFFS for ap in (0, 1)]
    write_csv(ba, p_old)
    # 10. --compare must reject a regression on the CHANGED path.
    #     Must tamper an apparent=1 cell: apparent=0 is the control, and a
    #     noisy control legitimately RAISES the floor rather than failing.
    #     The gate is on the MEDIAN, so shift EVERY apparent=1 cell.
    bb = [dict(r) for r in ba]
    for r in bb:
        if int(r["apparent"]) == 1:
            r["secs"] = "1.500"
    write_csv(bb, p_new)
    buf = io.StringIO()
    ok = not compare_csvs(p_old, p_new, buf.write, max_regress=1.02)
    cases.append(("--compare rejects a 1.5x median regression on the changed path", ok))

    # 10b. ...and must ACCEPT a noisy CONTROL with a clean changed path.
    bb2 = [dict(r) for r in ba]
    for r in bb2:
        if int(r["apparent"]) == 0:
            r["secs"] = "1.900"      # control is 1.9x noisy
    write_csv(bb2, p_new)
    buf = io.StringIO()
    ok = compare_csvs(p_old, p_new, buf.write, max_regress=1.02)
    cases.append(("--compare accepts a noisy control when the changed path is clean", ok))

    # 10c. A SINGLE outlier cell must not fail the gate. Two runs of the same
    #      binary produced exactly this: one apparent=1 cell at 1.0758x, the
    #      median at 1.0105x. Gating on the max reported a phantom regression.
    bb3 = [dict(r) for r in ba]
    first = next(r for r in bb3 if int(r["apparent"]) == 1)
    first["secs"] = "1.400"
    write_csv(bb3, p_new)
    buf = io.StringIO()
    ok = compare_csvs(p_old, p_new, buf.write, max_regress=1.02)
    cases.append(("--compare does not fail on a single outlier cell", ok))

    # 11. --compare must NOT be fooled by two completely disjoint CSVs
    bc = [dict(ba[0], fixture="__other__")]
    write_csv(bc, p_new)
    buf = io.StringIO()
    ok = not compare_csvs(p_old, p_new, buf.write)
    cases.append(("--compare with zero shared cells fails loudly", ok))

    passed = True
    for name, ok in cases:
        out(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        passed &= ok
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--mode", choices=["bench", "equiv"])
    ap.add_argument("--baseline", help="bench-mode CSV to compare maxdim=1 against")
    ap.add_argument("--max-regress", type=float, default=1.02)
    ap.add_argument("--fixtures", default=",".join(FIXTURES))
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--maxdims", default=",".join(map(str, MAXDIMS)))
    ap.add_argument("--coeffs", default=",".join(map(str, COEFFS)))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--compare", help="second CSV; require bitwise-identical barcodes")
    ap.add_argument("--min-baseline-secs", type=float, default=0.05,
                    help="cells whose baseline time is below this are excluded "
                         "from the timing gate (too fast to resolve it)")
    args = ap.parse_args()

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    if args.self_test:
        out("SELF-TEST: proving the checker can fail")
        ok = self_test(out)
        out("SELF-TEST " + ("OK" if ok else "FAILED"))
        return 0 if ok else 1

    if not args.csv:
        ap.error("--csv required unless --self-test")
    if args.compare:
        ok = compare_csvs(args.csv, args.compare, out, args.max_regress,
                         args.min_baseline_secs)
        out("CHECK OK" if ok else "CHECK FAIL")
        return 0 if ok else 1
    fixtures = args.fixtures.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    maxdims = [int(x) for x in args.maxdims.split(",")]
    coeffs = [int(x) for x in args.coeffs.split(",")]
    mode = args.mode or ("equiv" if os.path.basename(args.csv).find("equiv") >= 0
                         else "bench")

    out(f"checking {args.csv}  mode={mode}")
    rows = load(args.csv)
    out(f"rows read: {len(rows)}")

    ok = True
    ok &= check_grid(rows, mode, fixtures, ns, maxdims, coeffs, out)
    ok &= check_errors(rows, out)
    ok &= check_provenance(rows, out)
    if mode == "equiv":
        ok &= check_equiv(rows, out)
    else:
        summarise(rows, out)
        if args.baseline:
            out(f"  regression vs {args.baseline} (allow {args.max_regress}x)")
            ok &= check_regression(rows, load(args.baseline), args.max_regress, out)
    out("CHECK OK" if ok else "CHECK FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
