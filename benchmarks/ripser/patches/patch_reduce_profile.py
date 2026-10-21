#!/usr/bin/env python3
"""Patch a COPY of ripser++.cu with per-column timers for the reduction loop.

The point is to compare the three parts of the reduction against canns-lib's
own three-part split, which is measured per COLUMN (a handful of clock reads
per reducing column, so no distortion):

    scan   : enumerate the column's cofacets
    fill   : build the coboundary heap out of them, plus the first pop
    reduce : the reduction rounds (each re-enumerates the pivot's own
             coboundary and pushes it into the heap)

The accumulators are file-scope statics, printed and reset once per
dimension next to the existing "SUBMATRIX REDUCTION TIME" line.

The original ripser++.cu is NOT touched: this script works on a copy
(`rp_count`).
"""
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/sichaohe/ph_work/rp_count/ripser++.cu"

src = open(PATH).read()

# ---- 1. the accumulators ------------------------------------------------
anchor = "#define LOWER_DISTANCE_INDEX(i,j,n) (((i)*((i)-1)/2)+(j))"
assert anchor in src, "distance-index macro not found"
src = src.replace(
    anchor,
    anchor + """

// ---- REDUCE PROFILE (added by patch_reduce_profile.py) ------------------
// The reduction loop split into the same three parts canns-lib reports:
// scan (enumerate the cofacets), fill (build the heap, first pop) and
// reduce (the rounds). Double, so the clock resolution is not an issue.
static double g_rp_scan_s = 0.0;
static double g_rp_fill_s = 0.0;
static double g_rp_reduce_s = 0.0;
static long g_rp_cofacets = 0;
static long g_rp_cols = 0;
static long g_rp_rounds = 0;
static inline double rp_now() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
""",
    1,
)

# ---- 2. split init_coboundary_and_get_pivot_submatrix ------------------
old = """        bool check_for_emergent_pair= true;
        cofacet_entries.clear();
        simplex_coboundary_enumerator cofacets(simplex, dim, *this);
        while (cofacets.has_next()) {
            diameter_index_t_struct cofacet= cofacets.next();
            if (cofacet.diameter <= threshold) {
                cofacet_entries.push_back(cofacet);
                if (check_for_emergent_pair && (simplex.diameter == cofacet.diameter)) {
                    if(get_value_pivot_array_hashmap(cofacet.index, cmp)==-1) {
                        return cofacet;
                    }
                    check_for_emergent_pair= false;
                }
            }
        }
        for (auto cofacet : cofacet_entries) working_coboundary.push(cofacet);
        return get_pivot(working_coboundary);
    }"""
new = """        bool check_for_emergent_pair= true;
        cofacet_entries.clear();
        double _rp_t0= rp_now();
        simplex_coboundary_enumerator cofacets(simplex, dim, *this);
        while (cofacets.has_next()) {
            diameter_index_t_struct cofacet= cofacets.next();
            if (cofacet.diameter <= threshold) {
                cofacet_entries.push_back(cofacet);
                if (check_for_emergent_pair && (simplex.diameter == cofacet.diameter)) {
                    if(get_value_pivot_array_hashmap(cofacet.index, cmp)==-1) {
                        g_rp_scan_s += rp_now()-_rp_t0;
                        g_rp_cofacets += (long) cofacet_entries.size();
                        g_rp_cols += 1;
                        return cofacet;
                    }
                    check_for_emergent_pair= false;
                }
            }
        }
        g_rp_scan_s += rp_now()-_rp_t0;
        g_rp_cofacets += (long) cofacet_entries.size();
        g_rp_cols += 1;
        double _rp_t1= rp_now();
        for (auto cofacet : cofacet_entries) working_coboundary.push(cofacet);
        diameter_index_t_struct _rp_p= get_pivot(working_coboundary);
        g_rp_fill_s += rp_now()-_rp_t1;
        return _rp_p;
    }"""
assert old in src, "init_coboundary_and_get_pivot_submatrix body not found"
src = src.replace(old, new, 1)

# ---- 3. time the reduction rounds and count them ------------------------
old = """            if(index_column_to_add!=-1) {
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX

                        add_coboundary_reduction_submatrix(reduction_submatrix, index_column_to_add,
                                                           dim, working_reduction_column, working_coboundary);

                        pivot= get_pivot(working_coboundary);"""
new = """            if(index_column_to_add!=-1) {
                g_rp_rounds += 1;
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX

                        add_coboundary_reduction_submatrix(reduction_submatrix, index_column_to_add,
                                                           dim, working_reduction_column, working_coboundary);

                        pivot= get_pivot(working_coboundary);"""
assert old in src, "reduction round body not found"
src = src.replace(old, new, 1)

# ---- 4. time the whole reduction block ---------------------------------
old = """            pivot= init_coboundary_and_get_pivot_submatrix(column_to_reduce, working_coboundary, dim, cmp_pivots);

            while (true) {"""
new = """            double _rp_r0= rp_now();
            pivot= init_coboundary_and_get_pivot_submatrix(column_to_reduce, working_coboundary, dim, cmp_pivots);

            while (true) {"""
assert old in src, "reduction block start not found"
src = src.replace(old, new, 1)

# ---- 5. print and reset, next to the existing line ----------------------
old = """        std::cerr<<"SUBMATRIX REDUCTION TIME for dim "<< dim<<": "<<sw.ms()/1000.0<<"s"<<"\\n"<<std::endl;"""
new = """        std::cerr<<"SUBMATRIX REDUCTION TIME for dim "<< dim<<": "<<sw.ms()/1000.0<<"s"<<"\\n"<<std::endl;
        std::cerr<<"REDUCEPROF dim="<<dim<<" scan_s="<<g_rp_scan_s<<" fill_s="<<g_rp_fill_s
                 <<" reduce_s="<<g_rp_reduce_s<<" cols="<<g_rp_cols<<" rounds="<<g_rp_rounds
                 <<" cofacets="<<g_rp_cofacets<<std::endl;
        g_rp_scan_s = g_rp_fill_s = g_rp_reduce_s = 0.0;
        g_rp_cofacets = g_rp_cols = g_rp_rounds = 0;"""
assert old in src, "SUBMATRIX REDUCTION TIME print not found"
src = src.replace(old, new, 1)

open(PATH, "w").write(src)
print("patched %s" % PATH)
