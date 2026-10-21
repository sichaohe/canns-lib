#!/usr/bin/env python
"""How the reducing columns are actually reduced -- the counters that say.

WHY. The dim-2 reduction costs 15.03 s over only 298,670 columns that
reduce, i.e. 50 us PER COLUMN, which is ~150,000 cycles to find one pivot.
That is the largest single block of the gap against ripser++ (+3.57 s), so
"it is 3.57 s slower" is not an actionable statement. This probe answers
the question that IS actionable: how many coboundary entries does the
reduction touch per column?

The counters live in `walk_stats` (reduction.rs) and are reported on the
WALKSTATS line. They are gated behind `apparent-stats`, which ALSO enables
a quadratic DIAGNOSTIC (WALKCROSS scans the whole apparent list once per
reduced pivot). That diagnostic inflates the WALL of the instrumented
build massively -- 31.7 s to 81.9 s here -- but it does NOT touch the
reduction loop, so `loop_s` and `nonapparent_loop_s` remain comparable with
the uninstrumented build. The script asserts that rather than assuming it.

USAGE
    python loop_probe.py [--site ...] [--out loop_probe.csv]
"""

import argparse
import csv
import os
import re
import subprocess
import sys

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
CHILD = "/home/sichaohe/ph_work/h2h_child.py"
# The site built with `apparent-stats` (the counters) AND `gpu-phase-stats`.
SITE = "/home/sichaohe/canns-lib-dev/site_count2"

WALK_RE = re.compile(
    r"WALKSTATS dim=(?P<dim>\d+) membership_queries=(?P<queries>\d+) "
    r"hits=(?P<hits>\d+) heap_fills=(?P<heap_fills>\d+) "
    r"heap_entries=(?P<heap_entries>\d+) scan_cols=(?P<scan_cols>\d+) "
    r"slow_cols=(?P<slow_cols>\d+) cofacet_scan=(?P<cofacet_scan>\d+) "
    r"cofacet_fill=(?P<cofacet_fill>\d+) cofacet_add=(?P<cofacet_add>\d+) "
    r"reduction_adds=(?P<reduction_adds>\d+) matrix_entries=(?P<matrix_entries>\d+)"
)
STATS_RE = re.compile(
    r"STATS dim=(?P<dim>\d+) columns=(?P<columns>\d+) "
    r"apparent_pairs=(?P<apparent>\d+) .*?loop_s=(?P<loop_s>[\d.]+) "
    r"nonapparent_loop_s=(?P<nonapparent_loop_s>[\d.]+) "
    r"nonapparent_cols=(?P<nonapparent_cols>\d+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--out", default="loop_probe.csv")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    env = dict(os.environ, PYTHONPATH=args.site, CANNS_RIPSER_STATS="1")
    p = subprocess.run(
        ["/usr/bin/time", "-f", "__T__%e %M", PY, CHILD, args.site, "capped", "2"],
        capture_output=True, text=True, env=env, timeout=args.timeout,
    )
    if p.returncode != 0:
        print(p.stderr[-2000:], file=sys.stderr)
        raise SystemExit("probe run failed")

    walks = {}
    for m in WALK_RE.finditer(p.stderr):
        d = m.groupdict()
        d = {k: int(v) for k, v in d.items()}
        walks[d["dim"]] = d
    stats = {}
    for m in STATS_RE.finditer(p.stderr):
        d = m.groupdict()
        stats[int(d["dim"])] = d
    if not walks:
        raise SystemExit("no WALKSTATS line -- the counters are not compiled in")

    rows = []
    for dim in sorted(walks):
        w = walks[dim]
        s = stats.get(dim, {})
        fills = w["heap_fills"]
        entries = w["heap_entries"]
        loop_s = float(s.get("loop_s", 0.0))
        nonap_s = float(s.get("nonapparent_loop_s", 0.0))
        per_col = entries / fills if fills else 0.0
        cofacet_total = w["cofacet_scan"] + w["cofacet_fill"] + w["cofacet_add"]
        rows.append(dict(
            dim=dim,
            cofacet_scan=w["cofacet_scan"],
            cofacet_fill=w["cofacet_fill"],
            cofacet_add=w["cofacet_add"],
            cofacet_total=cofacet_total,
            reduction_adds=w["reduction_adds"],
            matrix_entries=w["matrix_entries"],
            ns_per_cofacet=round(nonap_s * 1e9 / cofacet_total, 2)
            if cofacet_total else 0.0,
            columns=int(s.get("columns", 0)),
            apparent_pairs=int(s.get("apparent", 0)),
            nonapparent_cols=int(s.get("nonapparent_cols", 0)),
            scan_cols=w["scan_cols"],
            slow_cols=w["slow_cols"],
            heap_fills=fills,
            heap_entries=entries,
            entries_per_heap_fill=round(per_col, 1),
            loop_s=loop_s,
            nonapparent_loop_s=nonap_s,
            ns_per_heap_entry=round(nonap_s * 1e9 / entries, 1) if entries else 0.0,
            ns_per_slow_column=round(nonap_s * 1e9 / w["slow_cols"], 1)
            if w["slow_cols"] else 0.0,
        ))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s" % args.out)
    for r in rows:
        print("dim %d: %d reducing columns, %d heap fills, %d entries "
              "(%.1f per fill), %.3f s  =>  %.1f ns per heap entry, "
              "%.1f us per slow column"
              % (r["dim"], r["slow_cols"], r["heap_fills"], r["heap_entries"],
                 r["entries_per_heap_fill"], r["nonapparent_loop_s"],
                 r["ns_per_heap_entry"], r["ns_per_slow_column"] / 1000.0))
        print("       cofacets: scan=%d fill=%d add=%d total=%d "
              "(%.2f ns per cofacet); reduction_adds=%d matrix_entries=%d"
              % (r["cofacet_scan"], r["cofacet_fill"], r["cofacet_add"],
                 r["cofacet_total"], r["ns_per_cofacet"],
                 r["reduction_adds"], r["matrix_entries"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
