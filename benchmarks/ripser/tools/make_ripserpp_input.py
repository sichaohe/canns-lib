#!/usr/bin/env python3
"""Generate the ripser++ input file for the ASA distance matrix.

WHY THIS FILE EXISTS. The head-to-head against ripser++ was invalid for
several rounds because the two sides were fed DIFFERENT filtrations, and
because the file format did not match the reader. Both traps are closed
here, and the checks at the bottom make a silent regression impossible.

THE FORMAT. ripser++'s `read_distance_matrix` (the `--format distance`
reader) does:

    for (int i = 0; std::getline(input_stream, line); ++i) {
        std::istringstream s(line);
        for (int j = 0; j < i && s >> value; ++j) { distances.push_back(value); ... }
    }

so it takes the FIRST `i` values of line `i`. The file must therefore be
the FULL n x n matrix, one row per line: line `i` holds d[i][0..i-1] in
its first `i` fields, and the reader picks exactly those. The diagonal is
at position `i` of line `i` and is skipped, so it may be anything (0 is
used here).

A lower-triangular-only file -- which is what the project's older
`convert_dm.py` wrote -- shifts every row by one and feeds ripser++ a
matrix that is not the one we meant to compare against. That produced a
"faster than expected" ripser++ whose column count (100,897,898) did not
match the 286,562,199 columns our own side was reducing.

THE FILTRATION. The ASA matrix has ~30% of its pairs at +inf (the pairs
that never join the fuzzy union graph). inf can never be printed, so the
pairs are capped at M = 2*max(finite) + 1. The threshold handed to
ripser++ is T = max(finite), and M > T, so every capped pair is strictly
above the threshold and can never be part of a simplex of diameter <= T.
The barcode is therefore the barcode of the FINITE part of the matrix --
the same filtration our own side computes when it is given the matrix
with the inf entries left alone.

NOTE: THRESHOLD IS T, NOT M. Passing `--threshold M` (the cap) instead
admits the capped pairs as real edges and grows the dim-2 column count
from 100,897,898 to 286,562,199 -- a different, much larger filtration.
The two sides must be told the same thing.

USAGE
    python make_ripserpp_input.py <dm.npy> <out.txt> [--mode full|lower]

    --mode full   (default) the format `--format distance` expects.
    --mode lower  the format `--format lower-distance` expects: line `i`
                  holds i values, with NO leading empty line.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npy", nargs="?", default="/home/sichaohe/ph_work/ph_dm_1200.npy")
    ap.add_argument("out", nargs="?", default="/tmp/asa1200_full.txt")
    ap.add_argument("--mode", choices=("full", "lower"), default="full")
    ap.add_argument("--fmt", default="%.10g", help="float format (default: %%.10g)")
    args = ap.parse_args()

    dm = np.load(args.npy)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        print("FAIL: expected a square distance matrix, got %r" % (dm.shape,), file=sys.stderr)
        return 1
    n = dm.shape[0]

    finite = np.isfinite(dm)
    if np.isnan(dm).any():
        print("FAIL: NaN in the distance matrix", file=sys.stderr)
        return 1
    t = float(dm[finite].max())
    cap = t * 2.0 + 1.0
    if not cap > t:
        print("FAIL: cap must exceed the threshold", file=sys.stderr)
        return 1

    capped = np.where(finite, dm, cap).astype(np.float64)
    np.fill_diagonal(capped, 0.0)

    with open(args.out, "w") as fh:
        if args.mode == "full":
            for i in range(n):
                fh.write(" ".join(args.fmt % v for v in capped[i]))
                fh.write("\n")
        else:
            for i in range(1, n):
                fh.write(" ".join(args.fmt % v for v in capped[i, :i]))
                fh.write("\n")

    size = os.path.getsize(args.out)
    print("mode=%s n=%d" % (args.mode, n))
    print("  finite      : %d / %d (%.1f%%)" % (finite.sum(), dm.size, 100.0 * finite.sum() / dm.size))
    print("  +inf        : %d (%.1f%%)  <- excluded from the filtration"
          % ((~finite).sum(), 100.0 * (~finite).sum() / dm.size))
    print("  threshold T : %.10g  (max finite)" % t)
    print("  cap M       : %.10g  (M > T, so capped pairs are above the threshold)" % cap)
    print("  wrote       : %s (%d bytes)" % (args.out, size))
    print("RUN: ripser++ --format distance --dim 2 --threshold %.10g --ratio 1.0 %s"
          % (t, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
