#!/usr/bin/env python
"""What exactly differs between the +inf-preserved and the capped input?

WHY THIS PROBE EXISTS. The head-to-head reports the same barcode md5 for
the +inf-preserved arm and the capped arm, and the +inf arm is 2.2x
faster. Those two facts together look like "the capped run computes the
same thing more slowly", and that reading would be WRONG in general:

  * capped input: every missing pair becomes the finite value `cap`, and
    `cap == thresh`, so `cap <= thresh` is TRUE. The missing pairs DO enter
    the complex. At scale `cap` the complex is therefore the FULL 2-skeleton
    -- every triple of vertices -- not the Rips complex of the real data.
  * +inf-preserved input: `inf <= thresh` is FALSE, so those simplices are
    never enumerated. The complex at the final scale is the Rips complex of
    the measured pairs only.

So the two runs compute DIFFERENT complexes, and the barcodes agree only
if the extra simplices happen to be invisible in the homology. This probe
measures whether that is the case HERE, and says exactly why.

It reports, for both input forms:
  * how many matrix entries are missing,
  * per dimension: total / essential / finite bars (the essential convention
    is applied IDENTICALLY to both sides -- this is the convention that has
    been got wrong three times in this project),
  * the dim-0 essential bars, which are the ones most likely to differ,
  * an element-wise comparison of the two bar arrays, reporting the first
    difference rather than only a hash.

USAGE
    python inf_form_probe.py [site] [--matrix path]
"""

import argparse
import sys

import numpy as np


def bars_of(site, d, cap, maxdim=2, coeff=47):
    sys.path.insert(0, site)
    from canns_lib.ripser import ripser
    r = ripser(d, maxdim=maxdim, thresh=cap, coeff=coeff,
               distance_matrix=True, progress_bar=False)["dgms"]
    out = []
    for x in r:
        a = np.asarray(x, dtype="float64")
        # Same convention on both sides: TOTAL = finite + essential.
        ess = int(np.isinf(a).any(axis=1).sum()) if a.size else 0
        out.append(dict(total=int(a.shape[0]), essential=ess,
                        finite=int(a.shape[0]) - ess,
                        arr=a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", nargs="?",
                    default="/home/sichaohe/canns-lib-dev/site_radix")
    ap.add_argument("--matrix", default="/home/sichaohe/ph_work/ph_dm_1200.npy")
    args = ap.parse_args()

    d0 = np.load(args.matrix)
    np.fill_diagonal(d0, 0.0)
    n = d0.shape[0]
    n_inf = int(np.isinf(d0).sum())
    cap = float(d0[np.isfinite(d0)].max()) * 2 + 1.0
    d_cap = np.where(np.isfinite(d0), d0, cap)

    print("matrix      %s  n=%d  shape=%s" % (args.matrix, n, d0.shape))
    print("missing     %d of %d upper-triangle entries are +inf (%.2f%%)"
          % (n_inf, n * (n - 1) // 2, 100.0 * n_inf / (n * (n - 1) // 2)))
    print("max finite  %.6f" % float(d0[np.isfinite(d0)].max()))
    print("cap         %.6f   (the value the missing pairs are given when capped)"
          % cap)
    print("threshold   %.6f   (passed to ripser in BOTH arms)" % cap)
    print()

    for name, d in (("+inf preserved", d0), ("capped", d_cap)):
        b = bars_of(args.site, d, cap)
        print("%-16s" % name, end="")
        for i, x in enumerate(b):
            print("  dim%d: total=%d essential=%d finite=%d"
                  % (i, x["total"], x["essential"], x["finite"]), end="")
        print()

    a = bars_of(args.site, d0, cap)
    b = bars_of(args.site, d_cap, cap)
    print()
    print("=== element-wise comparison (not just a hash) ===")
    same = True
    for i, (x, y) in enumerate(zip(a, b)):
        ax, ay = x["arr"], y["arr"]
        if ax.shape != ay.shape:
            print("dim %d: SHAPE differs %s vs %s" % (i, ax.shape, ay.shape))
            same = False
            continue
        # Compare as multisets of (birth, death) with inf handled explicitly.
        def key(rows):
            return sorted((float(r[0]), "inf" if np.isinf(r[1]) else "%.9g" % r[1])
                          for r in rows)
        kx, ky = key(ax), key(ay)
        if kx == ky:
            print("dim %d: IDENTICAL as multisets (%d bars)" % (i, len(kx)))
        else:
            same = False
            diff = [(p, q) for p, q in zip(kx, ky) if p != q]
            print("dim %d: DIFFER -- %d of %d bars differ; first 5:"
                  % (i, len(diff), len(kx)))
            for p, q in diff[:5]:
                print("        inf-preserved %s   vs   capped %s" % (p, q))
            # The essential bars are the interesting ones.
            ex = sorted((float(r[0]), float(r[1]))
                        for r in ax if np.isinf(r[1]).any())
            ey = sorted((float(r[0]), float(r[1]))
                        for r in ay if np.isinf(r[1]).any())
            print("        essential in inf-preserved: %s" % (ex[:6],))
            print("        essential in capped       : %s" % (ey[:6],))
    print()
    print("BOTH_ARMS_IDENTICAL_BARCODE" if same else "BARCODE_DIFFERS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
