#!/usr/bin/env python
"""A/B between two BUILDS of the library, in one interleaved batch.

Used for changes that are not behind a kill switch (the device-side change
that removed the host-side copy of the apparent population is one of them):
the only honest way to compare two builds on a shared machine is to run them
alternately inside one batch, because the machine's load moves baselines by
14% between batches.

The barcode md5 is checked in every run. A change that alters the answer is
not a speed change.

USAGE
    python two_site_ab.py --a SITE_A --b SITE_B --out ab.csv [--reps 3]
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
CHILD = "/home/sichaohe/ph_work/h2h_child.py"
REFERENCE_MD5 = (
    "ee1f5283091dd645a3dd53e6f1a7eac6:"
    "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
    "2d018d8fe9c1895b1b74f837af59ed20"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the baseline site")
    ap.add_argument("--b", required=True, help="the candidate site")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    arms = [("a", args.a), ("b", args.b)]
    expected = len(arms) * args.reps
    print("EXPECTED_CELLS %d (arms=%d x reps=%d)" % (expected, len(arms), args.reps),
          flush=True)
    print("A = %s\nB = %s" % (args.a, args.b), flush=True)

    rows = []
    for rep in range(1, args.reps + 1):
        for name, site in arms:
            env = dict(os.environ)
            env["PYTHONPATH"] = site
            la = float(open("/proc/loadavg").read().split()[0])
            p = subprocess.run(
                ["/usr/bin/time", "-f", "__T__%e %M", PY, CHILD, site, "capped", "2"],
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
            rows.append(dict(arm=name, site=site, rep=rep, wall=wall,
                             rss_kb=rss, bars=bars, md5=md5,
                             load_before="%.2f" % la))
            print("  rep%d %s wall=%7.3f rss=%6.2fGB md5=%s"
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
        v = [r["wall"] for r in rows if r["arm"] == name]
        if v:
            med[name] = statistics.median(v)
    if len(med) == 2:
        print("\nA median %.3f s, B median %.3f s, delta %+.3f s (%.3fx)"
              % (med["a"], med["b"], med["b"] - med["a"], med["b"] / med["a"]),
              flush=True)
    rss_a = max(r["rss_kb"] for r in rows if r["arm"] == "a")
    rss_b = max(r["rss_kb"] for r in rows if r["arm"] == "b")
    print("peak RSS: A %.2f GB, B %.2f GB (delta %+.2f GB)"
          % (rss_a / 1048576.0, rss_b / 1048576.0,
             (rss_b - rss_a) / 1048576.0), flush=True)
    bad = [r for r in rows if r["md5"] != REFERENCE_MD5]
    if bad:
        print("BARCODE MISMATCH in %d runs" % len(bad), flush=True)
        for r in bad:
            print("  %s" % r, flush=True)
        return 1
    print("barcode identical in all %d runs" % len(rows), flush=True)
    print("AB_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
