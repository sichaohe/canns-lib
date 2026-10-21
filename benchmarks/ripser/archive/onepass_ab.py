#!/usr/bin/env python
"""A/B for the one-pass cofacet enumeration, with a CSV.

The change under test: the reduction used to walk each reducing column's
coboundary TWICE -- once in `scan_coboundary_pivot`, and again in
`fill_coboundary_heap` when the scan found a pivot collision and the real
reduction had to happen. ripser++ walks it once (it keeps the cofacets in a
`cofacet_entries` vector and builds the heap from that).

The two enumerations are identical, so the heap content -- and therefore the
barcode -- is unchanged. Only the number of walks changes.

Both arms run in ONE interleaved batch, because the machine is shared and
its load moves baselines by 14% between batches.

USAGE
    python onepass_ab.py --out onepass_ab.csv [--reps 3]
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
SITE = "/home/sichaohe/canns-lib-dev/site_1pass"
CHILD = "/home/sichaohe/ph_work/h2h_child.py"
REFERENCE_MD5 = (
    "ee1f5283091dd645a3dd53e6f1a7eac6:"
    "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
    "2d018d8fe9c1895b1b74f837af59ed20"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    arms = [("one_pass", "1"), ("two_pass", "0")]
    expected = len(arms) * args.reps
    print("EXPECTED_CELLS %d (arms=%d x reps=%d)" % (expected, len(arms), args.reps),
          flush=True)

    rows = []
    for rep in range(1, args.reps + 1):
        for name, sw in arms:
            env = dict(os.environ)
            env["PYTHONPATH"] = SITE
            env["CANNS_RIPSER_ONE_PASS_COFACET"] = sw
            la = float(open("/proc/loadavg").read().split()[0])
            p = subprocess.run(
                ["/usr/bin/time", "-f", "__T__%e %M", PY, CHILD, SITE, "capped", "2"],
                capture_output=True, text=True, env=env, timeout=args.timeout,
            )
            m = re.search(r"__T__([\d.]+) (\d+)", p.stderr)
            wall = float(m.group(1)) if m else float("nan")
            rss = int(m.group(2)) if m else 0
            md5 = ""
            bars = ""
            for line in p.stdout.splitlines():
                if line.startswith("MD5 "):
                    md5 = line.split(" ", 1)[1].strip()
                elif line.startswith("BARS "):
                    bars = line.split(" ", 1)[1].strip()
            rows.append(dict(arm=name, rep=rep, switch=sw, wall=wall,
                             rss_kb=rss, bars=bars, md5=md5,
                             load_before="%.2f" % la))
            print("  rep%d %-9s wall=%7.3f rss=%6.2fGB md5=%s"
                  % (rep, name, wall, rss / 1048576.0, md5[:16]), flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows of %d expected)" % (args.out, len(rows), expected),
          flush=True)

    med = {}
    for name, _ in arms:
        ws = [r["wall"] for r in rows if r["arm"] == name]
        if ws:
            med[name] = statistics.median(ws)
    if len(med) == 2:
        print("\none_pass median %.3f s, two_pass median %.3f s, delta %+.3f s (%.3fx)"
              % (med["one_pass"], med["two_pass"],
                 med["one_pass"] - med["two_pass"],
                 med["one_pass"] / med["two_pass"]), flush=True)
    # The barcode must be identical in every arm and equal to the reference.
    bad = [r for r in rows
           if r["md5"] != REFERENCE_MD5 or r["bars"] != "1195|481|58"]
    if bad:
        print("BARCODE MISMATCH in %d runs" % len(bad), flush=True)
        for r in bad:
            print("  %s" % r, flush=True)
        return 1
    print("barcode identical in all %d runs and equal to the reference" % len(rows),
          flush=True)
    print("AB_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
