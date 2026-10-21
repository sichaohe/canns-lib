"""One measured run of canns-lib's ripser, for the final head-to-head.

Usage: h2h_child.py <site> <capped|inf_preserved> <maxdim>

Prints one machine-parseable block:
    WALL <seconds>          inner wall, excluding interpreter start and load
    BARS <n0|n1|n2>         bar count per dimension, TOTAL (finite + essential)
    MD5 <h0:h1:h2>          per-dimension barcode md5, bars sorted first
    H1TOP <a,b,c>           top-3 H1 lifetimes, rounded to 2 decimals

The md5 is the SAME construction the rest of the harness uses: each
dimension's bars are sorted before hashing. That makes it insensitive to the
ORDER of the bars, which is what we want here -- the order of the ASSEMBLED
COLUMNS is checked by the in-library SORTVERIFY differential test, not here.
"""

import hashlib
import sys
import time

import numpy as np

site, iform, maxdim = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, site)
from canns_lib.ripser import ripser  # noqa: E402

ASA = "/home/sichaohe/ph_work/ph_dm_1200.npy"

d = np.load(ASA)
np.fill_diagonal(d, 0.0)
cap = float(d[np.isfinite(d)].max()) * 2 + 1.0
if iform == "capped":
    d = np.where(np.isfinite(d), d, cap)

t0 = time.perf_counter()
r = ripser(d, maxdim=maxdim, thresh=cap, coeff=47, distance_matrix=True,
           progress_bar=False)["dgms"]
wall = time.perf_counter() - t0

md5 = ":".join(
    hashlib.md5(
        np.asarray(sorted(map(tuple, x)), dtype="float64").tobytes()
    ).hexdigest()
    for x in r
)
h1 = np.asarray(r[1], dtype="float64")
life = np.sort(h1[:, 1] - h1[:, 0])[::-1]
# Only finite lifetimes, on BOTH sides of the comparison: ripser++ prints an
# essential bar as `[0, )`, and an infinite lifetime would sit at the top of
# the sorted list and make the two sides incomparable for reasons that have
# nothing to do with the algorithm.
life = life[np.isfinite(life)]
top3 = ",".join("%.2f" % x for x in life[:3])

print("WALL %.3f" % wall)
print("BARS " + "|".join(str(len(x)) for x in r))
print("MD5 " + md5)
print("H1TOP " + top3)
