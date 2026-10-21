#!/usr/bin/env python3
"""Do two builds of canns-lib produce the SAME barcode?

WHY THIS SCRIPT EXISTS. The cofacet early exit changes how many DISTANCES
are read, not which simplices are found. That is a claim, and a claim about
the result is the kind that has to be checked rather than argued: a change
that returns a partial diameter for a rejected cofacet is correct only if
every caller filters on the threshold BEFORE it uses the diameter. Seven call
sites do, and this script checks the claim end to end -- the same input, two
binaries, the barcodes compared VALUE BY VALUE.

The COUNT of the bars is not enough. A change that alters the pairs while
keeping their number would survive a count comparison and still be a wrong
answer, so the values are hashed (sha256 over the printed pairs).

USAGE
    python barcode_equivalence.py \
        --old-site /home/sichaohe/canns-lib-dev/site_cpu_final \
        --new-site /home/sichaohe/canns-lib-dev/site_eearly \
        --out barcode_equivalence.csv
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_core

PY = bench_core.PY
CORPUS = bench_core.CORPUS
ASA = bench_core.ASA_MATRIX
O3 = os.path.join(bench_core.EXAMPLES, "o3_4096.point_cloud")

CHILD = r'''
import hashlib, os, sys, time
import numpy as np
site, path, kind, maxdim, coeff, thresh = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
    sys.argv[6])
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
    r = ripser(X, maxdim=maxdim, thresh=float(thresh), coeff=coeff,
               distance_matrix=False, progress_bar=False)["dgms"]
else:
    d = np.load(path)
    np.fill_diagonal(d, 0.0)
    r = ripser(d, maxdim=maxdim, thresh=float(thresh), coeff=coeff,
               distance_matrix=True, progress_bar=False)["dgms"]
h = hashlib.sha256()
counts = []
# `enumerate`, NOT `sorted`: `dgms` is a LIST of numpy arrays, and sorting it
# compares the arrays with each other -- which raises a broadcast error the
# moment two dimensions have different numbers of bars.
for dim, bars in enumerate(r):
    counts.append(str(len(bars)))
    for row in bars:
        h.update(("%.17g|%.17g;" % (row[0], row[1])).encode())
print("COUNTS " + "|".join(counts))
print("HASH " + h.hexdigest())
'''

FIELDS = ("input", "kind", "maxdim", "coeff", "thresh", "old_counts", "new_counts",
          "old_hash", "new_hash", "equal", "old_s", "new_s", "verdict")


def run(site, path, kind, maxdim, coeff, thresh, timeout):
    measured = bench_core.run_measured(
        [PY, "-c", CHILD, site, path, kind, str(maxdim), str(coeff), thresh],
        timeout=timeout, tag="beq_%s" % os.path.basename(path)[:40], keep=True)
    if measured.rc != 0:
        return "", "", measured.wall, "RC=%s" % measured.rc
    counts, hsh = "", ""
    for line in measured.stdout.splitlines():
        if line.startswith("COUNTS "):
            counts = line[7:].strip()
        elif line.startswith("HASH "):
            hsh = line[5:].strip()
    if not hsh:
        return counts, hsh, measured.wall, "NOHASH"
    return counts, hsh, measured.wall, "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-site", default="/home/sichaohe/canns-lib-dev/site_cpu_final")
    ap.add_argument("--new-site", default="/home/sichaohe/canns-lib-dev/site_eearly")
    ap.add_argument("--out", default="barcode_equivalence.csv")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--corpus-limit", type=int, default=0,
                    help="0 means every matrix in the corpus")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    inputs = [
        # The point cloud gets the threshold the o3 suite uses (1.4). At
        # threshold = inf the filtration is the full Vietoris-Rips complex of
        # 4096 points -- C(4096,3) = 11.4 billion triangles -- which is not a
        # benchmark, it is a hang, and it is what an earlier version of this
        # script did: the run was killed and the failure looked like a
        # difference in the barcode.
        ("o3_4096", O3, "point_cloud", 3, 2, "1.4"),
        ("asa_1200", ASA, "distance_matrix", 2, 47, "inf"),
    ]
    corpus = sorted(glob.glob(os.path.join(args.corpus, "*.npy")))
    if args.corpus_limit:
        corpus = corpus[:args.corpus_limit]
    for p in corpus:
        inputs.append((os.path.basename(p)[:-4], p, "distance_matrix", 2, 47, "inf"))

    rows = []
    bad = 0
    for name, path, kind, maxdim, coeff, thresh in inputs:
        if not os.path.exists(path):
            print("  SKIP %s (no %s)" % (name, path))
            continue
        oc, oh, os_, ov = run(args.old_site, path, kind, maxdim, coeff, thresh, args.timeout)
        nc, nh, ns, nv = run(args.new_site, path, kind, maxdim, coeff, thresh, args.timeout)
        equal = (oh == nh) and (oc == nc) and ov == "OK" and nv == "OK"
        if not equal:
            bad += 1
        rows.append({
            "input": name, "kind": kind, "maxdim": maxdim, "coeff": coeff,
            "thresh": thresh,
            "old_counts": oc, "new_counts": nc,
            "old_hash": oh, "new_hash": nh,
            "equal": "yes" if equal else "NO",
            "old_s": "%.3f" % os_, "new_s": "%.3f" % ns,
            "verdict": "%s/%s" % (ov, nv),
        })
        print("  %-28s %s  old=%.2fs new=%.2fs  %s"
              % (name, "SAME" if equal else "*** DIFFERENT ***", os_, ns,
                 "" if equal else (oh + " vs " + nh)), flush=True)

    bench_core.write_table(args.out, FIELDS, rows)

    print("\n%d inputs compared, %d differ -- wrote %s" % (len(rows), bad, args.out))
    if bad:
        print("\nFAIL: the two builds do not agree on %d input(s)" % bad)
        return 1
    print("\nPASS: the two builds produce byte-identical barcodes on every "
          "input -- the change is cheaper, not different")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
