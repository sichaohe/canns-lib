#!/usr/bin/env python3
"""Add a peak-heap-size probe to the ripser++ copy.

The question: our `working_coboundary` heap peaks at 11.5M entries (185 MB)
at dim 1 and 9.3M (148 MB) at dim 2, because a single column can take
10,243 / 21,423 reduction rounds and each round pushes the pivot's whole
1198-cofacet coboundary while popping one or two entries. If ripser++'s heap
stays small, the two implementations differ in the REDUCTION STRUCTURE, and
the difference is where the 2.64x lives. If it grows the same way, the
growth is inherent to the algorithm and the difference is in the constants.

Works on the copy in `rp_count`; the pristine `ripser++.cu` is never
touched.
"""
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/sichaohe/ph_work/ripserpp/rp_count/ripser++.cu"

src = open(PATH).read()

# ---- accumulators -------------------------------------------------------
anchor = "static inline double rp_now() {"
assert anchor in src, "rp_now() not found (patch_reduce_profile.py must run first)"
src = src.replace(
    anchor,
    """static size_t g_rp_heap_max = 0;
static size_t g_rp_heap_final = 0;
static size_t g_rp_rounds_max = 0;
static size_t g_rp_col_rounds = 0;
""" + anchor,
    1,
)

# ---- track the peak inside the reduction loop ---------------------------
old = """            if(index_column_to_add!=-1) {
                g_rp_rounds += 1;"""
new = """            if(index_column_to_add!=-1) {
                g_rp_rounds += 1;
                g_rp_col_rounds += 1;
                if (working_coboundary.size() > g_rp_heap_max) g_rp_heap_max = working_coboundary.size();"""
assert old in src, "reduction round body not found"
src = src.replace(old, new, 1)

# ---- close the per-column round counter and record the final heap size ---
# The reduction block ends right before the `} else {` that closes the
# "found a pair" branch; the column loop is in compute_pairs_plusplus, so
# the cleanest place to close the counter is where the next column starts.
# The `compute_pairs` (CPU-only) function has the same `column_to_reduce`
# line, so patch by LINE INDEX: take the occurrence that comes after the
# `compute_pairs_plusplus` definition.
lines = src.splitlines(True)
fn = next(i for i, l in enumerate(lines) if "void compute_pairs_plusplus(" in l)
hit = next(i for i, l in enumerate(lines)
           if i > fn and l.strip() == "working_coboundary;")
lines[hit + 1:hit + 1] = [
    "            if (g_rp_col_rounds > g_rp_rounds_max) g_rp_rounds_max = g_rp_col_rounds;\n",
    "            g_rp_col_rounds = 0;\n",
    "            g_rp_heap_final = working_coboundary.size();\n",
]
src = "".join(lines)

# ---- print ---------------------------------------------------------------
old = """        std::cerr<<"REDUCEPROF dim="<<dim<<" scan_s="<<g_rp_scan_s<<" fill_s="<<g_rp_fill_s
                 <<" reduce_s="<<g_rp_reduce_s<<" cols="<<g_rp_cols<<" rounds="<<g_rp_rounds
                 <<" cofacets="<<g_rp_cofacets<<std::endl;"""
new = """        std::cerr<<"REDUCEPROF dim="<<dim<<" scan_s="<<g_rp_scan_s<<" fill_s="<<g_rp_fill_s
                 <<" reduce_s="<<g_rp_reduce_s<<" cols="<<g_rp_cols<<" rounds="<<g_rp_rounds
                 <<" cofacets="<<g_rp_cofacets
                 <<" heap_max="<<g_rp_heap_max<<" col_rounds_max="<<g_rp_rounds_max<<std::endl;
        g_rp_heap_max = 0; g_rp_rounds_max = 0;"""
assert old in src, "REDUCEPROF print not found"
src = src.replace(old, new, 1)

open(PATH, "w").write(src)
print("patched %s with the heap-peak probe" % PATH)
