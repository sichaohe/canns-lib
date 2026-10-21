#!/usr/bin/env python
"""A/B for the process-level pinned-buffer cache, measured the way it pays.

WHY THIS NEEDS ITS OWN HARNESS. The cache only pays on the SECOND and later
calls inside one process -- the first call still has to page-lock its
buffers. A single-shot benchmark therefore cannot see the change at all,
and would report "no difference" for a change that saves 2.5 s per matrix
in a 97-matrix session.

So this runs the SAME matrix TWICE in one process and reports both calls.
The comparison that matters is within an arm: call #2 minus call #1.

  cache on   : call #2 should be ~2.5 s faster than call #1
  cache off  : call #2 should be about the same as call #1

USAGE
    python pinned_cache_ab.py --out pinned_cache_ab.csv [--reps 2]
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
SITE = "/home/sichaohe/canns-lib-dev/site_pcache"
MATRIX = "/home/sichaohe/ph_work/ph_dm_1200.npy"
REFERENCE_MD5 = (
    "ee1f5283091dd645a3dd53e6f1a7eac6:"
    "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
    "2d018d8fe9c1895b1b74f837af59ed20"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    arms = [("cache_on", {}), ("cache_off", {"CANNS_RIPSER_PINNED_CACHE": "0"})]
    expected = len(arms) * args.reps * 2
    print("EXPECTED_CELLS %d (arms=%d x reps=%d x calls=2)" % (expected, len(arms), args.reps),
          flush=True)

    rows = []
    for rep in range(1, args.reps + 1):
        for name, extra in arms:
            env = dict(os.environ)
            env["PYTHONPATH"] = SITE
            for k in ("CANNS_RIPSER_PINNED_CACHE",):
                env.pop(k, None)
            env.update(extra)
            la = float(open("/proc/loadavg").read().split()[0])
            p = subprocess.run(
                [PY, "-u", "/home/sichaohe/ph_work/two_calls.py", SITE, MATRIX, "2"],
                capture_output=True, text=True, env=env, timeout=args.timeout * 2,
            )
            calls = re.findall(r"CALL (\d+) wall=([\d.]+) md5=(\S+)", p.stdout)
            if len(calls) != 2:
                raise SystemExit("expected 2 CALL lines, got %d:\n%s"
                                 % (len(calls), p.stdout[-2000:] + p.stderr[-2000:]))
            for c, wall, md5 in calls:
                rows.append(dict(arm=name, rep=rep, call=int(c),
                                 wall=float(wall), md5=md5,
                                 load_before="%.2f" % la))
            print("  rep%d %-10s call1=%7.3f call2=%7.3f  delta=%+.3f"
                  % (rep, name, float(calls[0][1]), float(calls[1][1]),
                     float(calls[1][1]) - float(calls[0][1])), flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows of %d expected)" % (args.out, len(rows), expected),
          flush=True)

    summary = []
    for name, _ in arms:
        d = [r["wall"] for r in rows if r["arm"] == name and r["call"] == 2]
        c1 = [r["wall"] for r in rows if r["arm"] == name and r["call"] == 1]
        if d and c1:
            summary.append((name, statistics.median(c1), statistics.median(d)))
            print("  %-10s call1 median %.3f  call2 median %.3f  (call2-call1 %+.3f)"
                  % (name, statistics.median(c1), statistics.median(d),
                     statistics.median(d) - statistics.median(c1)), flush=True)
    bad = [r for r in rows if r["md5"] != REFERENCE_MD5]
    if bad:
        print("BARCODE MISMATCH in %d runs" % len(bad), flush=True)
        return 1
    print("barcode identical in all %d calls" % len(rows), flush=True)
    if len(summary) == 2:
        on = [s for s in summary if s[0] == "cache_on"][0]
        off = [s for s in summary if s[0] == "cache_off"][0]
        print("cache_on call2 %.3f vs cache_off call2 %.3f => %+.3f s"
              % (on[2], off[2], on[2] - off[2]), flush=True)
    print("AB_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
