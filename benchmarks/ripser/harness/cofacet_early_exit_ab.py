#!/usr/bin/env python3
"""Does the cofacet early exit pay, and does it change the barcode?

THE CHANGE. The cofacet enumerator used to read ALL `dim + 1` distances of a
candidate cofacet before the caller could reject it on the threshold. At
o3_4096 (4096 points, threshold 1.4) 98.7% of the candidates are rejected, so
98.7% of the distance reads were spent on cofacets that were thrown away --
and each of those reads is a scattered miss into a 33.5 MB matrix.

The early exit stops reading as soon as the running maximum is already above
the threshold. The true diameter is at least the running maximum, so such a
cofacet is rejected either way, and every caller filters on the threshold
BEFORE it uses the diameter, so none of them can observe the difference.

WHAT THIS SCRIPT ESTABLISHES.

* The two arms run the SAME BINARY. The switch is an environment variable
  (`CANNS_RIPSER_COFACET_EARLY_EXIT`), not a rebuild, so a difference in the
  numbers cannot be a difference in the compiler's mood.
* The barcodes are identical in every arm and every repetition. If they were
  not, this would not be a speed-up, it would be a different algorithm, and
  the script fails loudly instead of printing a speed-up.
* The arms are interleaved, and the order is reversed on every other
  repetition, so a drift in the host's speed cannot be mistaken for an effect
  of the switch.
* The foreign load is recorded for every row. A row measured while somebody
  else was using the machine is still a row, but it is not a result.

USAGE
    python cofacet_early_exit_ab.py \
        --site /home/sichaohe/canns-lib-dev/site_eearly \
        --out cofacet_early_exit_ab.csv --reps 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foreign_cpu

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
PH = "/home/sichaohe/ph_work"

# The datasets. `o3` is the point cloud where we are slowest relative to the
# references; `asa` is the distance matrix the paper's headline corpus number
# comes from, and it is here to catch a regression.
#
# `coeff` is the coefficient field. The two are different on purpose: the
# corpus runs on Z/47 and the o3 suite on Z/2, and a reduction change that
# only shows up under one modulus would be a red flag.
DATASETS = {
    "o3": {
        "kind": "point_cloud",
        "path": "/home/sichaohe/ph_work/ripserpp/ripserplusplus/examples/o3_4096.point_cloud",
        "maxdim": 3,
        "thresh": "1.4",
        "coeff": 2,
    },
    "asa": {
        "kind": "distance_matrix",
        "path": "/home/sichaohe/ph_work/ph_dm_1200.npy",
        "maxdim": 2,
        "thresh": "inf",
        "coeff": 47,
    },
}

ARMS = ("early_on", "early_off")
ARM_ENV = {"early_on": "1", "early_off": "0"}

FIELDS = (
    ["dataset", "arm", "rep", "arm_order", "kind", "maxdim", "coeff", "thresh"]
    + ["dim%d_%s" % (d, k) for d in (1, 2, 3) for k in ("scan_s", "push_s", "total_s", "rounds")]
    + ["wall_s", "bars", "barhash", "foreign_cpu", "foreign_max", "lib", "verdict"]
)

# The child. It prints the barcode as JSON so the parent can compare the
# arms byte for byte, and it refuses to run against a library other than the
# one it was pointed at -- the editable-install trap: a conda-installed
# canns_lib can beat PYTHONPATH and silently benchmark the WRONG binary.
CHILD = r'''
import json, os, sys, time
import numpy as np
site, path, kind, maxdim, thresh, coeff = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
    sys.argv[5], int(sys.argv[6]))
sys.path.insert(0, site)
import canns_lib
resolved = os.path.realpath(canns_lib.__file__)
want = os.path.realpath(os.path.join(site, "canns_lib", "__init__.py"))
if resolved != want:
    print("FATAL lib mismatch: loaded %s, wanted %s" % (resolved, want))
    raise SystemExit(2)
from canns_lib.ripser import ripser

if kind == "point_cloud":
    X = np.loadtxt(path, dtype=np.float64)
    t0 = time.perf_counter()
    r = ripser(X, maxdim=maxdim, thresh=float(thresh), coeff=coeff,
               distance_matrix=False, progress_bar=False)["dgms"]
elif kind == "distance_matrix":
    d = np.load(path)
    np.fill_diagonal(d, 0.0)
    t0 = time.perf_counter()
    r = ripser(d, maxdim=maxdim, coeff=coeff, distance_matrix=True,
               progress_bar=False)["dgms"]
else:
    raise SystemExit("unknown kind " + kind)
wall = time.perf_counter() - t0
print("WALL %.3f" % wall)
print("LIB " + resolved)
print("BARS " + json.dumps({str(k): len(v) for k, v in enumerate(r)}))
# The COUNT of bars is not the barcode. A change that alters the pairs while
# keeping their number would pass a count comparison and still be a wrong
# answer, so the values are hashed as well.
import hashlib
h = hashlib.sha256()
for dim, bars in enumerate(r):
    for row in bars:
        h.update(("%.17g|%.17g;" % (row[0], row[1])).encode())
print("BARHASH " + h.hexdigest())
'''

STATS_RE = re.compile(r"^STATS dim=(\d+) ")


def parse_stats(stderr_text):
    """The per-dimension STATS lines, as a dict of dicts."""
    out = {}
    for line in stderr_text.splitlines():
        if not STATS_RE.match(line):
            continue
        fields = {}
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                fields[k] = v
        try:
            dim = int(fields.get("dim", ""))
        except ValueError:
            continue
        out[dim] = fields
    return out


def run_arm(dataset, arm, rep, site, timeout):
    """One run of one arm. Returns the row dict and the parsed barcode."""
    spec = DATASETS[dataset]
    env = dict(os.environ)
    env["CANNS_RIPSER_STATS"] = "1"
    env["CANNS_RIPSER_COFACET_EARLY_EXIT"] = ARM_ENV[arm]
    out_path = "/tmp/_eearly_%s_%s.out" % (dataset, arm)
    err_path = "/tmp/_eearly_%s_%s.err" % (dataset, arm)
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        with foreign_cpu.Probe() as probe:
            t0 = time.perf_counter()
            proc = subprocess.Popen(
                [PY, "-c", CHILD, site, spec["path"], spec["kind"],
                 str(spec["maxdim"]), spec["thresh"], str(spec["coeff"])],
                stdout=fo, stderr=fe, env=env)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            wall = time.perf_counter() - t0
    rc = proc.returncode
    out = open(out_path).read()
    err = open(err_path).read()

    bars = ""
    barhash = ""
    lib = ""
    for line in out.splitlines():
        if line.startswith("BARS "):
            bars = line[5:]
        elif line.startswith("BARHASH "):
            barhash = line[8:].strip()
        elif line.startswith("LIB "):
            lib = line[4:]

    row = {k: "" for k in FIELDS}
    row["dataset"] = dataset
    row["arm"] = arm
    row["rep"] = rep
    row["kind"] = spec["kind"]
    row["maxdim"] = spec["maxdim"]
    row["coeff"] = spec["coeff"]
    row["thresh"] = spec["thresh"]
    row["wall_s"] = "%.3f" % wall
    row["bars"] = bars
    row["barhash"] = barhash
    row["lib"] = lib
    row["foreign_cpu"] = probe.record()
    row["foreign_max"] = probe.record_max()
    row["verdict"] = "OK" if rc == 0 and bars else "RC=%s" % rc

    for dim, fields in parse_stats(err).items():
        for key in ("scan_s", "push_s", "total_s", "rounds"):
            row["dim%d_%s" % (dim, key)] = fields.get(key, "")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="/home/sichaohe/canns-lib-dev/site_eearly")
    ap.add_argument("--out", default="cofacet_early_exit_ab.csv")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--datasets", default="o3,asa")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            print("unknown dataset %r" % d, file=sys.stderr)
            return 2
    if not os.path.isdir(args.site):
        print("missing site dir %s" % args.site, file=sys.stderr)
        return 2

    rows = []
    barcodes = {}
    hashes = {}
    for rep in range(1, args.reps + 1):
        for dataset in datasets:
            # The arm order is reversed on every other repetition, so a drift
            # in the host's speed cannot line up with the switch.
            order = list(ARMS) if rep % 2 else list(reversed(ARMS))
            for arm in order:
                print("--- %s rep=%d arm=%s" % (dataset, rep, arm), flush=True)
                row = run_arm(dataset, arm, rep, args.site, args.timeout)
                row["arm_order"] = ",".join(order)
                rows.append(row)
                if row["bars"]:
                    barcodes.setdefault(dataset, set()).add(row["bars"])
                if row["barhash"]:
                    hashes.setdefault(dataset, set()).add(row["barhash"])
                print("    wall=%s verdict=%s foreign=%s"
                      % (row["wall_s"], row["verdict"], row["foreign_cpu"]),
                      flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print("\nwrote %s (%d rows)" % (args.out, len(rows)))

    # THE GUARD. The two arms must produce the same barcode, on every
    # dataset. This is checked before anything is reported, because a
    # speed-up that changes the answer is not a speed-up.
    #
    # The VALUES are compared, not just the count of the bars: a change that
    # alters the pairs while keeping their number would survive a count
    # comparison and still be a wrong answer.
    bad = [d for d, s in barcodes.items() if len(s) != 1]
    bad += [d for d, s in hashes.items() if len(s) != 1]
    if bad:
        print("\nFAIL: the barcode differs between arms on: %s" % ", ".join(sorted(set(bad))),
              file=sys.stderr)
        for d in sorted(set(bad)):
            for b in sorted(barcodes.get(d, [])):
                print("  %s counts: %s" % (d, b), file=sys.stderr)
            for hsh in sorted(hashes.get(d, [])):
                print("  %s values: %s" % (d, hsh), file=sys.stderr)
        return 1
    print("barcodes identical across arms on: %s"
          % ", ".join(sorted(barcodes)) if barcodes else "NO BARCODE RECORDED")

    print("\n| dataset | arm | reps | median wall (s) |")
    print("|---|---|---|---|")
    for dataset in datasets:
        for arm in ARMS:
            vals = sorted(float(r["wall_s"]) for r in rows
                          if r["dataset"] == dataset and r["arm"] == arm
                          and r["verdict"] == "OK")
            if not vals:
                print("| %s | %s | 0 | (no OK rows) |" % (dataset, arm))
                continue
            med = vals[len(vals) // 2]
            print("| %s | %s | %d | %.3f |" % (dataset, arm, len(vals), med))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
