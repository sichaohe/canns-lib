#!/usr/bin/env python3
"""
Is the ASA matrix's 30% +inf actually costing us?

ripser++ has two ways to read the same matrix, measured in svd2.log:
    --sparse          : fills in the missing edges  -> 27.34 s  (maxdim=2)
    dense + capped    : missing edges stay absent  -> 11.28 s  (maxdim=2)
Same barcode, 2.42x apart. The difference is purely whether a +inf pair is
allowed to enter a simplex.

canns-lib currently runs with `thresh=inf`, and the cofacet filter is
`cofacet.get_diameter() <= self.threshold`, so `inf <= inf` is TRUE and every
missing edge does enter. That is the cost.

This measures canns-lib on the same matrix with the two threshold settings and
compares the barcodes bit for bit.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ["SITE"])
from canns_lib.ripser import ripser  # noqa: E402

ASA = "/home/sichaohe/ph_work/ph_dm_1200.npy"


def digest(dgm):
    out = []
    for bars in dgm:
        a = np.asarray(bars, dtype=np.float64)
        if a.size == 0:
            out.append("empty")
            continue
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
        out.append(f"{len(a)}:{float(a[:, 0].sum()):.6f}:{float(a[:, 1].sum()):.6f}")
    return " | ".join(out)


def main():
    d = np.load(ASA)
    np.fill_diagonal(d, 0.0)
    fin = np.isfinite(d)
    t_max = float(d[fin].max())
    cap = t_max * 2 + 1.0
    inf_frac = 1.0 - fin.sum() / (d.shape[0] ** 2 - d.shape[0])
    print(f"ASA 1200^2  finite pairs={fin.sum()}  +inf fraction={inf_frac:.1%}  "
          f"t_max={t_max:.4f}  cap={cap:.4f}")

    for name, thr in (("inf (today)", np.inf), ("capped (sparse semantics)", cap)):
        t0 = time.perf_counter()
        dgms = ripser(d, maxdim=2, thresh=thr, coeff=47,
                      distance_matrix=True, progress_bar=False)["dgms"]
        el = time.perf_counter() - t0
        print(f"  thresh={name:<28} wall={el:7.2f}s  bars="
              f"{[len(x) for x in dgms]}  {digest(dgms)}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
