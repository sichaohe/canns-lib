#!/usr/bin/env python3
"""Add push/pop timers to the ripser++ copy, matching canns-lib's split.

canns-lib's reduction rounds are split into
    push_s : add_coboundary_soa -- re-enumerating the pivot's coboundary and
             pushing it into the working coboundary heap
    pop_s  : get_pivot -- popping the pivot back out
so that the two implementations can be compared PART BY PART rather than
only in total. The same split is applied here:

    push_s : add_coboundary_reduction_submatrix
    pop_s  : get_pivot(working_coboundary)

Works on the copy in `rp_count`; the pristine `ripser++.cu` is never
touched.
"""
import io
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/sichaohe/ph_work/ripserpp/rp_count/ripser++.cu"
src = io.open(PATH, encoding="utf-8").read()

anchor = "static inline double rp_now() {"
assert anchor in src, "rp_now() not found (patch_reduce_profile.py must run first)"
src = src.replace(
    anchor,
    "static double g_rp_push_s = 0.0;\n"
    "static double g_rp_pop_s = 0.0;\n" + anchor,
    1,
)

old = """                        add_coboundary_reduction_submatrix(reduction_submatrix, index_column_to_add,
                                                           dim, working_reduction_column, working_coboundary);

                        pivot= get_pivot(working_coboundary);"""
new = """                        double _rp_p0= rp_now();
                        add_coboundary_reduction_submatrix(reduction_submatrix, index_column_to_add,
                                                           dim, working_reduction_column, working_coboundary);
                        g_rp_push_s += rp_now()-_rp_p0;

                        double _rp_q0= rp_now();
                        pivot= get_pivot(working_coboundary);
                        g_rp_pop_s += rp_now()-_rp_q0;"""
assert old in src, "reduction round body not found"
src = src.replace(old, new, 1)

old = """                 <<" cofacets="<<g_rp_cofacets
                 <<" heap_max="<<g_rp_heap_max<<" col_rounds_max="<<g_rp_rounds_max<<std::endl;"""
new = """                 <<" cofacets="<<g_rp_cofacets
                 <<" heap_max="<<g_rp_heap_max<<" col_rounds_max="<<g_rp_rounds_max
                 <<" push_s="<<g_rp_push_s<<" pop_s="<<g_rp_pop_s<<std::endl;
        g_rp_push_s = g_rp_pop_s = 0.0;"""
assert old in src, "REDUCEPROF print not found"
src = src.replace(old, new, 1)

io.open(PATH, "w", encoding="utf-8").write(src)
print("patched %s with push/pop timers" % PATH)
