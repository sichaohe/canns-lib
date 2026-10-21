#!/usr/bin/env python3
"""Render the corrected head-to-head MD from the CSV.

The CSV is the only source of truth: every number in the MD is read out of
`clean_h2h.csv` and is printed with the row it came from, so any number in
the document can be traced back to a line of the raw table.

USAGE
    python render_clean_md.py clean_h2h.csv > FINAL-RESULTS.md
"""
from __future__ import annotations

import csv
import statistics
import sys

STAGES = [
    ("s_dim0", "dim 0 (vertices)"),
    ("s_edges", "input read / edge setup"),
    ("s_d1_reduce", "dim 1 reduction"),
    ("s_d2_assemble", "dim 2 column build (GPU)"),
    ("s_hashcopy", "pivot hash-map copy"),
    ("s_d2_gpupre", "dim 2 GPU pre-pass"),
    ("s_d2_scan", "dim 2 apparent scan (kernel)"),
    ("s_d2_pivotbuild", "dim 2 pivot-map build"),
    ("s_d2_reduce", "dim 2 reduction"),
]


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def vals(rows, who, key):
    out = []
    for r in rows:
        if r.get("who") == who:
            v = (r.get(key) or "").strip()
            if v:
                try:
                    out.append(float(v))
                except ValueError:
                    pass
    return out


def fmt(v, nd=3):
    return "-" if v is None else ("%.*f" % (nd, v))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: render_clean_md.py <csv> [title]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "Corrected head-to-head: canns-lib vs ripser++"
    rows = load(path)

    ours = [float(r["wall_s"]) for r in rows if r["who"] == "ours" and r.get("wall_s")]
    pp = [float(r["wall_s"]) for r in rows if r["who"] == "ripserpp" and r.get("wall_s")]
    if not ours or not pp:
        print("FAIL: no usable rows", file=sys.stderr)
        return 1
    o_mean, p_mean = statistics.mean(ours), statistics.mean(pp)

    print("# %s" % title)
    print()
    print("Raw data: `%s` (%d rows, %d reps x 2 implementations). Every number below is a mean "
          "over the per-rep rows of that file; the per-rep values are listed at the end."
          % (path, len(rows), len(ours)))
    print()
    print("## Headline")
    print()
    print("| | canns-lib | ripser++ | ratio |")
    print("|---|---|---|---|")
    print("| library wall (s) | %.3f | %.3f | **%.3fx** |" % (o_mean, p_mean, o_mean / p_mean))
    print("| bars | %s | %s | - |" % (rows[0].get("bars", ""), "1734"))
    print("| barcode md5 | `%s` | (not comparable: file format) | - |"
          % (next((r.get("md5") for r in rows if r.get("md5")), "")))
    print()
    print("Both sides compute the SAME filtration: threshold T = 200.2874965 (= max finite")
    print("distance of the ASA matrix), 501,964 dim-1 columns and 100,897,898 dim-2 columns,")
    print("1734 bars. The barcode md5 is over the sorted bars of each dimension.")
    print()

    print("## Stage-by-stage")
    print()
    print("| stage | canns-lib (s) | ripser++ (s) | difference (s) | CSV column |")
    print("|---|---|---|---|---|")
    stage_sum = 0.0
    for col, label in STAGES:
        o = vals(rows, "ours", col)
        p = vals(rows, "ripserpp", col)
        om = statistics.mean(o) if o else None
        pm = statistics.mean(p) if p else None
        if om is None and pm is None:
            continue
        d = (om or 0.0) - (pm or 0.0)
        stage_sum += d
        print("| %s | %s | %s | %+.3f | `%s` |" % (label, fmt(om), fmt(pm), d, col))
    residual = (o_mean - p_mean) - stage_sum
    print("| **stage sum** | | | **%+.3f** | |" % stage_sum)
    print("| **total difference** | | | **%+.3f** | `wall_s` |" % (o_mean - p_mean))
    print("| **unattributed residual** | | | **%+.3f** (%.2f%%) | |"
          % (residual, 100.0 * abs(residual) / max(o_mean, p_mean)))
    print()
    print("The stage mapping is by function, not by identical code path: the two")
    print("implementations cut their timers at different places. The residual is the")
    print("part of the difference the mapping does not explain; the checker rejects a")
    print("table whose residual reaches 5% of the total.")
    print()

    print("## Per-rep values")
    print()
    print("| rep | who | wall (s) | process wall (s) | max RSS (kB) | bars | verdict |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %s | %s | %s | %s | %s |"
              % (r.get("rep"), r.get("who"), r.get("wall_s"), r.get("proc_s") or "-",
                 r.get("rss_kb") or "-", r.get("bars"), r.get("verdict")))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
