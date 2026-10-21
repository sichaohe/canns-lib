#!/usr/bin/env python3
"""
Three ways to read the same 30%-inf matrix, all on the real ASA matrix.

  A  thresh=inf, dense            what canns-lib does today     43.66 s (instrumented)
  B  thresh=cap, dense            missing edges stay absent     17.46 s
  C  COO of the finite pairs      uses the EXISTING ripser_dm_sparse path

B and C are the two ways to express "sparse semantics". C is the one the
library already implements (scipy-sparse input), so if it is competitive the
auto-switch is just "detect a mostly-missing matrix and hand it to the path
that is already there".
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

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


def timed(label, dm, **kw):
    t0 = time.perf_counter()
    dgms = ripser(dm, maxdim=2, coeff=47, distance_matrix=True,
                  progress_bar=False, **kw)["dgms"]
    el = time.perf_counter() - t0
    print(f"  {label:<34} wall={el:7.2f}s  bars={[len(x) for x in dgms]}")
    print(f"      {digest(dgms)}")
    sys.stdout.flush()
    return el


def main():
    d = np.load(ASA)
    np.fill_diagonal(d, 0.0)
    fin = np.isfinite(d)
    t_max = float(d[fin].max())
    cap = t_max * 2 + 1.0
    iu = np.triu_indices_from(d, 1)
    rows, cols = iu
    vals = d[iu]
    m = np.isfinite(vals)
    print(f"ASA 1200^2  finite upper pairs={m.sum()}/{len(vals)} "
          f"({m.sum()/len(vals):.1%})  t_max={t_max:.4f}  cap={cap:.4f}")

    timed("A thresh=inf  (today)", d, thresh=np.inf)
    timed("B thresh=cap  (dense)", d, thresh=cap)

    coo = sp.coo_matrix(
        (vals[m].astype(np.float32),
         (rows[m].astype(np.int32), cols[m].astype(np.int32))),
        shape=d.shape)
    timed("C COO of finite pairs (sparse)", coo, thresh=np.inf)


if __name__ == "__main__":
    main()
