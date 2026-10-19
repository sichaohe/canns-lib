// CUDA kernels for canns-lib's ripser backend, compiled by nvcc in build.rs.
//
// WHY A .cu FILE AND NOT AN NVRTC STRING. The kernels used to live as Rust
// string literals passed to `cudarc::nvrtc::compile_ptx_with_opts` at runtime.
// That works, but it constrains the source:
//   * NVRTC ships no system headers, so `<cstdint>` is unavailable and the
//     kernel had to hand-declare `typedef long long i64;`;
//   * no `__forceinline__`;
//   * `cudarc::safe::Params` is a tuple capped at 12 arguments
//     (`impl_launch!` only goes up to `[A..L]`).
// nvcc removes all three limits, which is what makes it possible to translate
// ripser++'s kernels (`ripser++.cu`, MIT) closely instead of paraphrase them.
//
// WHY BOTH A CUBIN AND A PTX. `cuModuleLoadData` sniffs the format and takes
// either, but they fail in different places:
//   * cubin is machine code for one exact architecture: loaded directly, no
//     JIT, therefore independent of the driver's CUDA version;
//   * ptx is JIT-ed by the driver at load time and is rejected outright if it
//     was produced by a newer toolkit (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).
// So: cubin preferred, ptx as the fallback for "cubin's arch does not match
// this machine". The same split is used in flashbrain2, where it was the
// difference between working and not on an A6000 (driver 550 / CUDA 12.4,
// toolkit 12.5).
//
// ⚠️ The NVRTC fallback is retained in gpu.rs / gpu_assembly.rs, so building
// with the `gpu` feature still does not require nvcc -- the prebuilt images are
// simply empty and the runtime falls back to the CPU path it already had.


// `i64`/`u64` used to be hand-declared because NVRTC ships no system
// headers; nvcc has `<cstdint>`.
//
// The include is vestigial -- the aliases below do not need it -- but NVRTC
// rejects ANY `#include` of a system header, so it must be skipped there.
// `__CUDACC_RTC__` is defined by NVRTC and by nothing else.
#ifndef __CUDACC_RTC__
#include <cstdint>
#endif
using i64 = long long;
using u64 = unsigned long long;



// Helpers live at namespace scope: C++ has no nested function definitions, and
// putting them inside __global__ yields a parse error whose cascade looks like
// twenty unrelated failures.

__device__ i64 bc(const i64 *b, i64 n_max, int k, i64 n) {
    return b[(i64)k * n_max + n];
}

__device__ i64 unrank_vertex(const i64 *b, i64 n_max, i64 idx, int k, i64 hi) {
    i64 lo = k - 1, top = hi;
    while (lo + 1 < top) {
        i64 mid = lo + ((top - lo) >> 1);
        if (bc(b, n_max, k, mid) <= idx) lo = mid; else top = mid;
    }
    return lo;
}

__device__ void unrank(const i64 *b, i64 n_max, int n, i64 idx, int dim, i64 *out) {
    i64 idxr = idx;
    int k = dim + 1;
    i64 hi = n;
    for (int t = 0; t < dim + 1; ++t) {
        i64 v = unrank_vertex(b, n_max, idxr, k, hi);
        out[dim - t] = v;
        idxr -= bc(b, n_max, k, v);
        hi = v;
        k -= 1;
    }
}

__device__ float diam_of(const float *dist, int n, const i64 *v, int nv) {
    float d = 0.0f;
    for (int a = 0; a < nv; ++a)
        for (int c = a + 1; c < nv; ++c)
            d = fmaxf(d, dist[v[a] * n + v[c]]);
    return d;
}

// One output record of the apparent pre-pass, 16 bytes, laid out to match
// canns-lib's `DiameterEntryT` EXACTLY:
//
//   bytes 0..8  : the packed entry, (index << 8) | (coefficient & 0xFF)
//   bytes 8..12 : the apparent partner's diameter (the "death")
//   bytes 12..16: padding
//
// The host reads the page-locked buffer IN PLACE as `&[DiameterEntryT]`, so
// the 4.6 GB population is never copied to the host a second time. The
// layout is pinned by a Rust unit test (`apparent_entry_layout`), so a
// compiler that reorders the struct fails the build loudly instead of
// silently corrupting the data.
//
// The packing must match `EntryT::new`: the index is shifted left by
// NUM_COEFFICIENT_BITS (8) and the coefficient occupies the low 8 bits.
// `get_index` is an arithmetic shift, so the -1 sentinel survives.
struct ApparentRec { long long entry; float diameter; };

// Two bits per column, packed into u64 words:
//   bit 0: this column is an apparent pair (it never reduces)
//   bit 1: this column ALSO contributes a bar (death > diameter * ratio)
//
// The host uses this to skip the apparent columns without reading the
// 4.6 GB apparent population at all -- which is what made the apparent
// fast path cost 1.55 s for 286,981,060 columns that produced ZERO bars.
//
// The ratio test has to happen HERE, because the host would otherwise have
// to read the column's own diameter (another 4.6 GB) to evaluate it.
extern "C" __global__ void detect_apparent(
    const i64   *binomial,
    i64 n_max,
    const float *dist,
    const i64   *cols,
    i64 n_cols, int n, int dim, float threshold, int modulus, float ratio,
    ApparentRec *out_rec, unsigned long long *out_flags)
{
    for (i64 gid = blockIdx.x * (i64)blockDim.x + threadIdx.x;
         gid < n_cols; gid += (i64)gridDim.x * blockDim.x) {

        i64 idx = cols[gid];
        i64 sv[8];
        unrank(binomial, n_max, n, idx, dim, sv);
        float target = diam_of(dist, n, sv, dim + 1);

        i64 idx_below = idx, idx_above = 0, v = n - 1;
        int k = dim + 1;
        i64 cofacet_index = -1;
        float cofacet_diam = 0.0f;
        int cofacet_coeff = 0;
        bool found = false;

        while (v >= (i64)k && !found) {
            while (bc(binomial, n_max, k, v) <= idx_below) {
                idx_below -= bc(binomial, n_max, k, v);
                idx_above += bc(binomial, n_max, k + 1, v);
                v -= 1;
                k -= 1;
            }
            float d = target;
            for (int a = 0; a <= dim; ++a)
                d = fmaxf(d, dist[v * n + sv[a]]);
            i64 ci = idx_above + bc(binomial, n_max, k + 1, v) + idx_below;

            // Coefficient mirrors SimplexCoboundaryEnumerator::next(): sign
            // flips on odd k, base is 1, then a mod.
            int sign = (k & 1) != 0 ? (modulus - 1) : 1;
            int cc = sign % modulus;
            if (cc < 0) cc += modulus;

            if (d <= threshold && d == target) {
                cofacet_index = ci;
                cofacet_diam = d;
                cofacet_coeff = cc;
                found = true;
            }
            v -= 1;
        }

        i64 result = -1;
        if (found) {
            i64 cv[8];
            unrank(binomial, n_max, n, cofacet_index, dim + 1, cv);
            i64 best = -1;
            int nv2 = dim + 2;
            for (int rem = 0; rem < nv2; ++rem) {
                float fd = 0.0f;
                for (int a = 0; a < nv2; ++a) {
                    if (a == rem) continue;
                    for (int c = a + 1; c < nv2; ++c) {
                        if (c == rem) continue;
                        fd = fmaxf(fd, dist[cv[a] * n + cv[c]]);
                    }
                }
                if (fd == cofacet_diam) {
                    i64 fi = 0;
                    int rank = 1;
                    for (int a = 0; a < nv2; ++a) {
                        if (a == rem) continue;
                        fi += bc(binomial, n_max, rank, cv[a]);
                        rank += 1;
                    }
                    if (best == -1 || fi < best) best = fi;
                }
            }
            if (best == idx) result = cofacet_index;
        }

        // Pack exactly like `EntryT::new(index, coefficient)`. The shift is
        // done on the unsigned value so the bit pattern of a negative index
        // (the -1 sentinel) is the same as Rust's wrapping `<<`.
        out_rec[gid].entry = ((unsigned long long)result << 8)
                           | ((unsigned long long)(cofacet_coeff) & 0xFFull);
        out_rec[gid].diameter = found ? cofacet_diam : -1.0f;

        // The flags. `result != -1` means the column is an apparent pair;
        // the bar flag additionally applies the host's ratio filter, which
        // the kernel can evaluate because it has BOTH the column's own
        // diameter (`target`) and the partner's (`cofacet_diam`).
        unsigned long long bits = 0;
        if (result != -1) {
            bits |= 1ull;
            float death = cofacet_diam;
            if (death > target * ratio) bits |= 2ull;
        }
        if (bits) {
            unsigned int w = (unsigned int)(gid >> 5);
            unsigned int b = (unsigned int)((gid & 31) * 2);
            atomicOr(&out_flags[w], bits << b);
        }
    }
}


// The assembly kernels reuse `bc` / `unrank_vertex` / `unrank` from the detect
// section above: identical bodies, and in one translation unit a second
// definition is a hard compile error.
//
// One output record, 16 bytes: matches canns-lib's DiameterIndexT and halves
// the number of pointer arguments (cudarc's Params tuple tops out at 12).
struct Col { float d; long long i; };

__device__ bool is_pivot(const u64 *pk, i64 n_pivots, i64 key) {
    i64 lo = 0, hi = n_pivots - 1;
    while (lo <= hi) {
        i64 mid = (lo + hi) >> 1;
        u64 v = pk[mid];
        if (v == (u64)key) return true;
        if (v < (u64)key) lo = mid + 1; else hi = mid - 1;
    }
    return false;
}

// Pass 1: per-simplex counts for both outputs. `all` counts every cofacet
// under threshold (next_simplices); `cols` counts the non-pivot subset
// (columns_to_reduce).
extern "C" __global__ void count_columns(
    const i64 *simplices, i64 n_simplices,
    const i64 *binomial, i64 n_max,
    const float *dist, i64 n_dim_packed, float threshold,
    const u64 *pivot_keys, i64 n_pivots,
    unsigned int *counts_all, unsigned int *counts_cols)
{
    int n = (int)(n_dim_packed >> 32);
    int dim = (int)((n_dim_packed >> 8) & 0xFFFFFFLL);  // same packing as write_columns
    i64 gid = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    i64 stride = (i64)gridDim.x * blockDim.x;

    for (; gid < n_simplices; gid += stride) {
        i64 sidx = simplices[gid];
        i64 verts[16];
        unrank(binomial, n_max, n, sidx, dim, verts);
        float sdiam = 0.0f;
        for (int a = 0; a <= dim; ++a)
            for (int c = a + 1; c <= dim; ++c)
                sdiam = fmaxf(sdiam, dist[verts[a] * n + verts[c]]);

        i64 idx_below = sidx, idx_above = 0, v = n - 1;
        int k = dim + 1;
        unsigned int ca = 0, cc = 0;

        while (v >= (i64)k && bc(binomial, n_max, k, v) > idx_below) {
            i64 vv = v, kk = k, ib = idx_below, ia = idx_above;
            while (bc(binomial, n_max, kk, vv) <= ib) {
                ib -= bc(binomial, n_max, kk, vv);
                ia += bc(binomial, n_max, kk + 1, vv);
                vv -= 1;
                kk -= 1;
            }
            float cd = sdiam;
            for (int a = 0; a <= dim; ++a)
                cd = fmaxf(cd, dist[vv * n + verts[a]]);
            i64 ci = ia + bc(binomial, n_max, kk + 1, vv) + ib;
            v -= 1;

            if (cd <= threshold) {
                ca += 1;
                if (!is_pivot(pivot_keys, n_pivots, ci)) cc += 1;
            }
        }
        counts_all[gid] = ca;
        counts_cols[gid] = cc;
    }
}

// Pass 2: write both outputs at their prefix-sum offsets.
extern "C" __global__ void write_columns(
    const i64 *simplices, i64 n_simplices,
    const i64 *offsets,           // [all_off, col_off] interleaved per simplex
    const i64 *binomial, i64 n_max,
    const float *dist, i64 n_dim_packed, float threshold,
    const u64 *pivot_keys, i64 n_pivots,
    Col *out_all, Col *out_cols)
{
    // n_dim_packed: n in the high 32, dim in bits 8..31, keep_next in bit 0.
    // cudarc caps Params at 12 arguments, so flags ride inside this word
    // rather than as a 13th.
    int n = (int)(n_dim_packed >> 32);
    int dim = (int)((n_dim_packed >> 8) & 0xFFFFFFLL);
    bool keep_next = (n_dim_packed & 1LL) != 0;
    i64 gid = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    i64 stride = (i64)gridDim.x * blockDim.x;

    for (; gid < n_simplices; gid += stride) {
        i64 sidx = simplices[gid];
        i64 verts[16];
        unrank(binomial, n_max, n, sidx, dim, verts);
        float sdiam = 0.0f;
        for (int a = 0; a <= dim; ++a)
            for (int c = a + 1; c <= dim; ++c)
                sdiam = fmaxf(sdiam, dist[verts[a] * n + verts[c]]);

        i64 idx_below = sidx, idx_above = 0, v = n - 1;
        int k = dim + 1;
        i64 sa = offsets[2 * gid], sc = offsets[2 * gid + 1];

        while (v >= (i64)k && bc(binomial, n_max, k, v) > idx_below) {
            i64 vv = v, kk = k, ib = idx_below, ia = idx_above;
            while (bc(binomial, n_max, kk, vv) <= ib) {
                ib -= bc(binomial, n_max, kk, vv);
                ia += bc(binomial, n_max, kk + 1, vv);
                vv -= 1;
                kk -= 1;
            }
            float cd = sdiam;
            for (int a = 0; a <= dim; ++a)
                cd = fmaxf(cd, dist[vv * n + verts[a]]);
            i64 ci = ia + bc(binomial, n_max, kk + 1, vv) + ib;
            v -= 1;

            if (cd <= threshold) {
                // Skipping the write is NOT optional: the buffer was sized
                // accordingly, and writing anyway is an out-of-bounds access
                // (CUDA_ERROR_ILLEGAL_ADDRESS).
                if (keep_next) {
                    out_all[sa].d = cd;
                    out_all[sa].i = ci;
                    sa += 1;
                }
                if (!is_pivot(pivot_keys, n_pivots, ci)) {
                    out_cols[sc].d = cd;
                    out_cols[sc].i = ci;
                    sc += 1;
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// GPU radix sort over the assembled columns.
//
// WHY. The assembled columns must end up ordered by (diameter DESC, index
// ASC) -- exactly the order the host-side `par_sort_unstable_by` produced. On
// the host that costs 2.43 s at ASA scale: it moves 4.6 GB with a comparison
// sort, and `total_cmp` on f32 is not a plain `<`. A radix sort moves the same
// bytes 16 times with NO comparisons, which is the one thing a GPU is good at.
// ripser++ does exactly this (`thrust::sort` on the device, then a single D2H
// copy); we cannot use thrust -- it is host-side C++ and this crate is
// driver-API only, with no cudart anywhere -- so the kernels are here.
//
// KEY LAYOUT (u64):
//   bits 63..32 : !sortable_u32(diameter)  -> ASCENDING key = DESCENDING diameter
//   bits 31..0  : (u32)index                -> ties broken by ASCENDING index
// `sortable_u32` is exactly total_cmp's ordering (negatives inverted, sign bit
// forced for non-negatives). The diameter half is inverted so an ASCENDING key
// sort yields DESCENDING diameter.
//
// The host MUST guarantee index < 2^32, or the low half silently truncates and
// the tie-break becomes wrong; the host checks that and falls back to the
// host-side sort otherwise.
//
// STABILITY. An LSD radix sort is only correct if every pass is stable, and
// the LAST pass (the diameter half) is exactly the one that decides the order
// among equal diameters. So the per-tile ranking below is computed exactly:
// each thread owns a contiguous run of items, counts them per digit, and the
// per-digit prefix over threads comes from a shared exclusive scan.
//
// FOUR BITS PER PASS, not eight or sixteen. The per-thread digit counts are
// the memory bottleneck: 256 threads x 16 bins x 4 B = 16 KB of shared memory,
// and 16 passes cover the 64-bit key. Sixteen bins per pass would need
// 256 x 65536 x 4 B = 64 MB of shared memory, which does not exist.
//
// TILE: 4096 items, 256 threads, 16 items per thread, BLOCKED (thread t owns
// items [t*16, t*16+16) of the tile) so "preceding within the tile" is well
// defined and the per-tile histograms compose into a global stable order.
#define RADIX_BITS 4
#define RADIX_BINS (1 << RADIX_BITS)
#define RADIX_TILE 4096
#define RADIX_THREADS 256
#define RADIX_IPT (RADIX_TILE / RADIX_THREADS)

__device__ __forceinline__ u64 col_sort_key(const Col &c) {
    unsigned int b = __float_as_uint(c.d);
    // total_cmp ascending key: invert for negatives, force the sign bit for
    // non-negatives.
    unsigned int s = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    // Invert, so a LARGER diameter yields a SMALLER key half.
    return ((u64)(~s) << 32) | (unsigned int)((unsigned long long)c.i);
}

// Per-tile histogram of the digit at `shift`: hist[tile * RADIX_BINS + bin].
extern "C" __global__ void radix_hist4(
    const Col *in, long long n, int shift, unsigned int *hist)
{
    __shared__ unsigned int s[RADIX_BINS];
    int tid = threadIdx.x;
    for (int i = tid; i < RADIX_BINS; i += RADIX_THREADS) s[i] = 0;
    __syncthreads();

    long long tile_start = (long long)blockIdx.x * RADIX_TILE;
    long long tile_end = tile_start + RADIX_TILE;
    if (tile_end > n) tile_end = n;
    for (long long i = tile_start + tid; i < tile_end; i += RADIX_THREADS) {
        unsigned int d = (unsigned int)((col_sort_key(in[i]) >> shift) & (RADIX_BINS - 1));
        atomicAdd(&s[d], 1u);
    }
    __syncthreads();
    for (int i = tid; i < RADIX_BINS; i += RADIX_THREADS)
        hist[(size_t)blockIdx.x * RADIX_BINS + i] = s[i];
}

// Exclusive scan along the tile axis, per bin, plus the cross-bin base.
// `hist` is rewritten in place: hist[tile*16+bin] becomes the offset of that
// tile's first element within its bin. `bin_offset[bin]` becomes the global
// base of that bin.
extern "C" __global__ void radix_scan4(
    unsigned int *hist, int n_tiles, unsigned int *bin_offset)
{
    __shared__ unsigned int totals[RADIX_BINS];
    int b = threadIdx.x;
    if (b >= RADIX_BINS) return;
    unsigned int running = 0;
    for (int t = 0; t < n_tiles; ++t) {
        unsigned int v = hist[(size_t)t * RADIX_BINS + b];
        hist[(size_t)t * RADIX_BINS + b] = running;
        running += v;
    }
    totals[b] = running;
    __syncthreads();
    unsigned int base = 0;
    for (int i = 0; i < b; ++i) base += totals[i];
    bin_offset[b] = base;
}

// Stable scatter. Every item's position inside its tile is computed exactly:
//   rank_in_tile = (count of same-digit items in threads before mine)
//                + (count of same-digit items in my thread before this one)
extern "C" __global__ void radix_scatter4(
    const Col *in, Col *out, long long n, int shift,
    const unsigned int *hist, const unsigned int *bin_offset)
{
    __shared__ unsigned int tcount[RADIX_THREADS][RADIX_BINS];
    __shared__ unsigned int toff[RADIX_BINS][RADIX_THREADS];

    int tid = threadIdx.x;
    long long tile_start = (long long)blockIdx.x * RADIX_TILE;
    long long tile_end = tile_start + RADIX_TILE;
    if (tile_end > n) tile_end = n;

    for (int b = tid; b < RADIX_BINS; b += RADIX_THREADS)
        for (int t = 0; t < RADIX_THREADS; ++t) tcount[t][b] = 0;
    __syncthreads();

    long long my_base = tile_start + (long long)tid * RADIX_IPT;
    unsigned int local[RADIX_BINS];
    for (int b = 0; b < RADIX_BINS; ++b) local[b] = 0;
    for (int k = 0; k < RADIX_IPT; ++k) {
        long long i = my_base + k;
        if (i < tile_end) {
            unsigned int d = (unsigned int)((col_sort_key(in[i]) >> shift) & (RADIX_BINS - 1));
            local[d] += 1u;
        }
    }
    for (int b = 0; b < RADIX_BINS; ++b) tcount[tid][b] = local[b];
    __syncthreads();

    // Per-digit exclusive prefix over the thread axis. 16 threads (one per
    // digit) each walk the 256 per-thread counts.
    if (tid < RADIX_BINS) {
        unsigned int run = 0;
        for (int t = 0; t < RADIX_THREADS; ++t) {
            toff[tid][t] = run;
            run += tcount[t][tid];
        }
    }
    __syncthreads();

    unsigned int seen[RADIX_BINS];
    for (int b = 0; b < RADIX_BINS; ++b) seen[b] = 0;
    for (int k = 0; k < RADIX_IPT; ++k) {
        long long i = my_base + k;
        if (i < tile_end) {
            unsigned int d = (unsigned int)((col_sort_key(in[i]) >> shift) & (RADIX_BINS - 1));
            unsigned int rank_in_tile = toff[d][tid] + seen[d];
            seen[d] += 1u;
            unsigned int base = bin_offset[d] + hist[(size_t)blockIdx.x * RADIX_BINS + d];
            out[base + rank_in_tile] = in[i];
        }
    }
}
