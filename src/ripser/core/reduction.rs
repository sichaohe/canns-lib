use crate::ripser::matrix::sparse::{CompressedSparseMatrix, OptimizedSparseMatrix};
use crate::ripser::matrix::traits::{
    CofacetEnumerator, DistanceMatrix, EdgeProvider, HasCofacets, VertexBirth,
};
use crate::ripser::types::{
    CoefficientT, DiameterEntryT, DiameterIndexT, IndexT, PivotVal, ValueT, WorkingT,
};
use crate::ripser::utils::field::multiplicative_inverse_vector;
use crate::ripser::utils::{modp, BinomialCoeffTable};
use rustc_hash::FxHashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// How many columns' apparent-pair detections are computed before the
/// sequential walk consumes them.
///
/// Bounded on purpose. Detection results are one `DiameterEntryT` per column
/// (~16 B), so a whole-dimension pre-pass at maxdim=2 / n=1200 would be
/// C(1200,3) ~= 2.9e8 columns ~= 4.6 GB. Chunking turns that into a ~1 MB
/// working set at the default while still giving P2 enough columns per chunk to
/// amortise thread-pool overhead.
///
/// Override with CANNS_RIPSER_APPARENT_CHUNK.
const DEFAULT_APPARENT_CHUNK: usize = 1 << 16;

fn apparent_chunk_size() -> usize {
    std::env::var("CANNS_RIPSER_APPARENT_CHUNK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_APPARENT_CHUNK)
}

// Per-cofacet instrumentation for the apparent-pair scan.
//
// Behind the `apparent-stats` cargo feature, and compiled to nothing when the
// feature is off. Two earlier shapes of this counter were both wrong:
//   * a process-wide AtomicU64 -- ~76M contended fetch_adds across 16 threads
//     collapsed a measured 6.75x speedup to 1.28x;
//   * a OnceLock check + thread_local -- cheap, but still ~6% on the very path
//     it was measuring.
// The measurement finally ran from the feature-gated no-op below, which costs
// zero in a normal release build.

#[cfg(feature = "apparent-stats")]
mod apparent_stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    // Atomics, not thread_local. The pre-pass runs on rayon workers while the
    // STATS line is printed from the thread that called in, so per-thread
    // counters read back as ZERO and the line reported `cofacets_per_col=0.000`
    // for every run -- a confidently wrong number, which is worse than none.
    // Relaxed ordering is all that is needed: these are counters, never used to
    // order anything, and the cost is only paid in builds that ask for them.
    static COFACETS: AtomicU64 = AtomicU64::new(0);
    static COLUMNS: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub fn count_cofacet() {
        COFACETS.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn count_column() {
        COLUMNS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn cofacets_per_column() -> f64 {
        COFACETS.load(Ordering::Relaxed) as f64 / COLUMNS.load(Ordering::Relaxed).max(1) as f64
    }
}

#[cfg(not(feature = "apparent-stats"))]
mod apparent_stats {
    #[inline(always)]
    pub fn count_cofacet() {}
    #[inline(always)]
    pub fn count_column() {}
    pub fn cofacets_per_column() -> f64 {
        0.0
    }
}

/// Whether to spread the apparent-pair pre-pass across rayon threads.
///
/// Defaults to ON when the `parallel` feature is compiled in, because the
/// pre-pass is a pure function of (column, dim). Set
/// CANNS_RIPSER_APPARENT_PAR=0 to force the sequential loop (used by the
/// benchmark harness to A/B the two without rebuilding).
#[cfg(feature = "parallel")]
fn apparent_parallel_enabled() -> bool {
    std::env::var("CANNS_RIPSER_APPARENT_PAR")
        .map(|v| {
            !matches!(
                v.trim(),
                "0" | "false" | "False" | "no" | "No" | "off" | "OFF"
            )
        })
        .unwrap_or(true)
}

/// Minor/major page faults and user/system jiffies for this process.
///
/// Reads `/proc/self/stat`, which needs no privilege -- unlike `perf`, which
/// `perf_event_paranoid = 4` blocks entirely on this machine (see report
/// S22.4/S23.1). Two reads bracket the walk, so the cost is nil.
///
/// Fields are 1-indexed per proc(5): 10 minflt, 12 majflt, 14 utime, 15 stime.
/// Field 2 is the command name in parentheses and may contain spaces, so
/// parsing starts after the last ')'.
fn proc_counters() -> (u64, u64, u64, u64) {
    let Ok(s) = std::fs::read_to_string("/proc/self/stat") else {
        return (0, 0, 0, 0);
    };
    let Some(idx) = s.rfind(')') else {
        return (0, 0, 0, 0);
    };
    let rest: Vec<&str> = s[idx + 1..].split_whitespace().collect();
    // rest[0] is field 3 (state), so field N is rest[N - 3].
    let get = |n: usize| -> u64 { rest.get(n - 3).and_then(|v| v.parse().ok()).unwrap_or(0) };
    (get(10), get(12), get(14), get(15))
}

#[cfg(not(feature = "parallel"))]
fn apparent_parallel_enabled() -> bool {
    false
}

/// Column count at which the apparent-pair path reliably pays for itself.
///
/// Derived from the 112-cell grid, not chosen by taste. The payoff is monotone
/// in column count: 1.01-1.05x below 2^17, 1.18x in [2^18, 2^19), 2.08x at 1e6+.
/// 2^18 is the lowest power of two where every measured cell at or above it was
/// a win; the two cells between 2^17 and 2^18 also won, but that is a thin
/// sample and the bucket just below it contains the losses.
const APPARENT_MIN_COLUMNS: usize = 1 << 18;

/// Tri-state env read: Some(true)/Some(false) when the caller said so, None when
/// they did not. Kept separate from the threshold so the benchmark harness and
/// the interleaved A/B arms can still force either path.
///
/// An unrecognised value falls through to the threshold rather than to "off": a
/// typo must not silently disable a measured 1.93x win.
fn env_forced(name: &str) -> Option<bool> {
    parse_forced(std::env::var(name).ok().as_deref())
}

/// Pure half of `env_forced`, so it can be tested without mutating the
/// process environment. `std::env::set_var` is not testable here: Rust guards
/// it against concurrent mutation and panics when another thread of the same
/// process -- of which this test binary has many, they all own rayon pools --
/// is reading the environment at the same moment.
fn parse_forced(value: Option<&str>) -> Option<bool> {
    match value.map(str::trim) {
        Some("1" | "true" | "True" | "yes" | "Yes") => Some(true),
        Some("0" | "false" | "False" | "no" | "No" | "") => Some(false),
        _ => None,
    }
}

/// Pick the array the sharded apparent table is built from.
///
/// Pulled out as a free function purely so it can be unit-tested WITHOUT a GPU.
/// The regression this exists to prevent: collapsing the two production sources
/// into one unconditional choice builds the table from an array that the GPU
/// path never fills, yielding an EMPTY table. Nothing crashes -- `get` simply
/// returns None for every key, so every apparent pivot is lost and the barcode
/// is quietly wrong while wall time looks better.
#[inline]
fn sharded_pivot_source<'a>(
    use_gpu_pre: bool,
    gpu_pre: &'a [DiameterEntryT],
    apparent_pre: &'a [DiameterEntryT],
) -> &'a [DiameterEntryT] {
    if use_gpu_pre {
        gpu_pre
    } else {
        apparent_pre
    }
}

fn apparent_requested(columns: usize) -> bool {
    env_forced("CANNS_RIPSER_APPARENT").unwrap_or(columns >= APPARENT_MIN_COLUMNS)
}

/// Same threshold, same rationale: the sharded build's saving is proportional
/// to the number of apparent entries, which is exactly what grows with the
/// column count. Below the threshold the build costs more than it saves, and at
/// small column counts it is not built at all (`use_gpu_pre` requires
/// GPU_MIN_COLUMNS). Measured on the ASA matrix: 39.24 s with, 50.69 s without.
fn sharded_pivot_requested(columns: usize) -> bool {
    env_forced("CANNS_RIPSER_SHARDED_PIVOT").unwrap_or(columns >= APPARENT_MIN_COLUMNS)
}

#[cfg(test)]
mod sharded_source_tests {
    use super::*;

    fn filled(n: usize) -> Vec<DiameterEntryT> {
        (0..n)
            .map(|i| DiameterEntryT::new(i as f32, i as IndexT + 1, 1))
            .collect()
    }

    /// The exact regression: choosing the wrong array must be visible here, not
    /// only in a full GPU benchmark run.
    #[test]
    fn each_path_gets_the_array_it_actually_filled() {
        let gpu = filled(64);
        let apparent = filled(64);
        let empty: Vec<DiameterEntryT> = Vec::new();

        assert_eq!(
            sharded_pivot_source(true, &gpu, &apparent).len(),
            64,
            "GPU path must read gpu_pre, NOT apparent_pre"
        );
        assert_eq!(
            sharded_pivot_source(false, &gpu, &apparent).len(),
            64,
            "CPU path must read apparent_pre, NOT gpu_pre"
        );
        // The shape of the original bug: on the GPU path apparent_pre is empty,
        // so an unconditional `apparent_pre` silently yields zero columns.
        assert_eq!(
            sharded_pivot_source(true, &gpu, &empty).len(),
            64,
            "an empty apparent_pre must not be able to hijack the GPU path"
        );
        assert_eq!(
            sharded_pivot_source(false, &empty, &apparent).len(),
            64,
            "an empty gpu_pre must not be able to hijack the CPU path"
        );
    }

    #[test]
    fn wrong_source_is_detectably_empty() {
        let empty: Vec<DiameterEntryT> = Vec::new();
        assert!(sharded_pivot_source(true, &empty, &empty).is_empty());
    }
}

#[cfg(test)]
mod apparent_gate_tests {
    use super::*;

    fn requested(forced: Option<&str>, columns: usize) -> bool {
        parse_forced(forced).unwrap_or(columns >= APPARENT_MIN_COLUMNS)
    }

    #[test]
    fn unset_falls_back_to_the_column_threshold() {
        assert!(!requested(None, 0));
        assert!(!requested(None, APPARENT_MIN_COLUMNS - 1));
        assert!(requested(None, APPARENT_MIN_COLUMNS));
        assert!(requested(None, 286_562_199));
    }

    #[test]
    fn explicit_values_override_the_threshold_in_both_directions() {
        for on in ["1", "true", "True", "yes", "Yes", " yes "] {
            assert_eq!(parse_forced(Some(on)), Some(true), "{on:?}");
            assert!(requested(Some(on), 1), "{on:?} should force on");
            assert!(requested(Some(on), 1 << 30));
        }
        for off in ["0", "false", "False", "no", "No", " 0 ", ""] {
            assert_eq!(parse_forced(Some(off)), Some(false), "{off:?}");
            assert!(!requested(Some(off), 1), "{off:?} should force off");
            assert!(!requested(Some(off), 1 << 30));
        }
    }

    #[test]
    fn unrecognised_values_fall_back_to_the_threshold() {
        // A typo must not silently disable a measured 1.93x win, nor silently
        // enable a path the caller believed they had turned off.
        for junk in ["maybe", "2", "on", "off", "TRUE", "-1"] {
            assert_eq!(parse_forced(Some(junk)), None, "{junk:?}");
            assert!(requested(Some(junk), APPARENT_MIN_COLUMNS));
            assert!(!requested(Some(junk), 1));
        }
    }
}

/// One lookup with the semantics of the single `FxHashMap` it replaces.
///
/// The pivot set has two populations. Apparent pairs contribute one entry per
/// enumerated column (286,263,529 on the ASA matrix); real reductions contribute
/// only as many as there are reduced columns (298,670). Splitting them lets the
/// bulk population live in a table that is built in parallel before the walk
/// starts (`ShardedPivotTable`, ~4.0 s) instead of inserted one at a time
/// during it (~22.1 s).
///
/// The resolution order reproduces the original map exactly. In the original
/// every insert is `FxHashMap::insert`, i.e. last writer wins, so at query
/// column `j` a key `p` resolves to whichever of the two populations was written
/// LAST:
///   * `reduced` only ever holds entries written at an earlier column, so it is
///     visible without any further check;
///   * the apparent table is bulk-preloaded, so an entry is visible only once
///     `pos < j` -- that is what reproduces "inserted before j";
///   * if both populations hold the key, the larger position wins, which is
///     precisely the overwrite the original map performed. (Measured: the two
///     sets are disjoint, `collide_with_apparent = 0`, so this arm never fires
///     on the ASA matrix -- it is here so the split cannot silently change
///     semantics on some other input.)
#[derive(Clone, Copy)]
pub struct PivotLookup<'a> {
    reduced: &'a FxHashMap<IndexT, PivotVal>,
    apparent: Option<&'a crate::ripser::pivot_sharded::ShardedPivotTable>,
}

impl<'a> PivotLookup<'a> {
    #[inline]
    pub fn new(
        reduced: &'a FxHashMap<IndexT, PivotVal>,
        apparent: Option<&'a crate::ripser::pivot_sharded::ShardedPivotTable>,
    ) -> Self {
        Self { reduced, apparent }
    }

    #[inline]
    pub fn get(&self, key: IndexT, before: usize) -> Option<PivotVal> {
        let apparent = match self.apparent {
            Some(t) => t.get(key),
            None => None,
        };
        let reduced = self.reduced.get(&key).copied();
        match (apparent, reduced) {
            (Some(a), Some(r)) => {
                // Strict `>` on the tie, because within ONE column the walk
                // records the apparent pivot first (the fast path at the top of
                // the loop body) and the reduction pivot last (at the bottom),
                // so at equal positions the reduction entry is the later write.
                // A column cannot actually be both -- the apparent path
                // `continue`s -- but the rule costs nothing to get right.
                if a.pos() >= before {
                    Some(r)
                } else if a.pos() > r.pos() {
                    Some(a)
                } else {
                    Some(r)
                }
            }
            (Some(a), None) => {
                if a.pos() < before {
                    Some(a)
                } else {
                    None
                }
            }
            (None, r) => r,
        }
    }

    #[inline]
    pub fn contains(&self, key: IndexT, before: usize) -> bool {
        self.get(key, before).is_some()
    }
}

/// Query counters for the reduction walk, behind `apparent-stats`.
///
/// The question they answer: how many of the 286.3M apparent pivots are ever
/// looked up? Only NON-apparent columns reach the lookup sites (apparent ones
/// `continue` first), so the queried set is bounded by the coboundaries of the
/// ~298,670 reduced columns. If that bound is far below 286M, then most of the
/// map is write-only and the writes are the thing worth removing.
#[cfg(feature = "apparent-stats")]
mod walk_stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static QUERIES: AtomicU64 = AtomicU64::new(0);
    pub static HITS: AtomicU64 = AtomicU64::new(0);
    // HOW the reducing columns are reduced.
    //
    // The dim-2 reduction costs 15.06 s over only 298,670 columns that
    // actually reduce -- 50 us EACH, which is ~150,000 cycles per column.
    // The fast path (`scan_coboundary_pivot`) does a linear scan and finds
    // the pivot without a heap; the SLOW path (`fill_coboundary_heap` +
    // `get_pivot`) materialises the column's WHOLE coboundary into a
    // BinaryHeap first. A dim-2 column is a triangle and its coboundary is
    // the tetrahedra containing it -- up to n-3 of them.
    //
    // These counters answer the question that decides whether the slow path
    // is the problem: how many columns take it, and how many entries does it
    // push in total. One atomic add per column (NOT per heap push) keeps the
    // overhead in the noise; the counts, not the timings, are the point.
    pub static HEAP_FILLS: AtomicU64 = AtomicU64::new(0);
    pub static HEAP_ENTRIES: AtomicU64 = AtomicU64::new(0);
    /// One increment per coboundary cancellation: two entries with the same
    /// simplex index whose coefficients are combined. This is the Z/p
    /// counterpart of ripser++'s Z/2 "pop the pair and drop it", and it is
    /// the only place in the reduction where a MODULAR REDUCTION happens
    /// per heap entry.
    pub static CANCELLATIONS: AtomicU64 = AtomicU64::new(0);
    pub static SCAN_COLUMNS: AtomicU64 = AtomicU64::new(0);
    pub static SLOW_COLUMNS: AtomicU64 = AtomicU64::new(0);
    #[inline]
    pub fn count_query() {
        QUERIES.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn count_hit() {
        HITS.fetch_add(1, Ordering::Relaxed);
    }
    /// One call per coboundary cancellation.
    #[inline]
    pub fn note_cancel() {
        CANCELLATIONS.fetch_add(1, Ordering::Relaxed);
    }
    /// One call per column that materialises its coboundary heap.
    #[inline]
    pub fn note_heap_fill(entries: usize) {
        HEAP_FILLS.fetch_add(1, Ordering::Relaxed);
        HEAP_ENTRIES.fetch_add(entries as u64, Ordering::Relaxed);
    }
    /// One call per column that takes the linear-scan fast path.
    #[inline]
    pub fn note_scan_column() {
        SCAN_COLUMNS.fetch_add(1, Ordering::Relaxed);
    }
    /// One call per column that ends up in the full reduction.
    #[inline]
    pub fn note_slow_column() {
        SLOW_COLUMNS.fetch_add(1, Ordering::Relaxed);
    }
    /// How many bars the APPARENT fast path actually produced.
    ///
    /// The fast path walks every column on the host to find the apparent
    /// pairs, and at ASA scale that is 286,263,529 iterations -- 1.55 s of
    /// the dim-2 reduction. It only pays off if some of those columns
    /// contribute a bar. The ratio filter is `death > diameter * ratio`
    /// with ratio = 1.0, and for an apparent pair (sigma, tau) the death is
    /// tau's diameter, which is >= sigma's diameter -- so the filter only
    /// passes for the pairs whose death is INFINITE. This counter says how
    /// many there are, which decides whether the whole walk is worth
    /// keeping.
    pub static APPARENT_BARS: AtomicU64 = AtomicU64::new(0);
    pub static APPARENT_COLS: AtomicU64 = AtomicU64::new(0);
    #[inline]
    pub fn note_apparent_bar() {
        APPARENT_BARS.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn note_apparent_col() {
        APPARENT_COLS.fetch_add(1, Ordering::Relaxed);
    }
    /// Nanoseconds spent in the pivot LOOKUPS.
    ///
    /// The lookups are only ~734K calls at dim 2 (299,340 `contains` plus
    /// 434,515 `get`), so a clock read per call costs ~18 ms and the
    /// measurement is clean. This is the part of the reduction that the
    /// cofacet counters do NOT cover, and it is where the remaining gap
    /// against ripser++ has to be: ripser++ does a BINARY SEARCH over a
    /// 4.6 GB sorted array per lookup while we do a hash probe into a
    /// 13.1 GB table, so the per-lookup cost is not obviously in our
    /// favour.
    pub static LOOKUP_NS: AtomicU64 = AtomicU64::new(0);
    #[inline]
    pub fn note_lookup_ns(ns: u64) {
        LOOKUP_NS.fetch_add(ns, Ordering::Relaxed);
    }
    /// How many cofacets the reduction's ENUMERATIONS visited, in total.
    ///
    /// Two numbers, because the reduction used to visit the coboundary
    /// TWICE per reducing column: once in `scan_coboundary_pivot` and again
    /// in `fill_coboundary_heap`. ripser++ visits it once (it keeps the
    /// cofacets in a `cofacet_entries` vector and builds the heap from
    /// that), so the ratio between these two counters is the size of the
    /// redundant work.
    ///
    /// Counted in bulk -- one atomic add per column, not per cofacet --
    /// because an atomic per cofacet would cost more than the thing it
    /// measures.
    pub static COFACET_SCAN: AtomicU64 = AtomicU64::new(0);
    pub static COFACET_FILL: AtomicU64 = AtomicU64::new(0);
    /// The coboundary pushes that happen AFTER a pivot is found: for the
    /// pivot column's own coboundary, and for the coboundary of every
    /// non-zero entry of the pivot column in the reduction matrix. This is
    /// the part of the reduction that `fill_coboundary_heap` does NOT see,
    /// and it is the part ripser++ serves from its pre-assembled compressed
    /// submatrix.
    pub static COFACET_ADD: AtomicU64 = AtomicU64::new(0);
    pub static REDUCTION_ADDS: AtomicU64 = AtomicU64::new(0);
    pub static MATRIX_ENTRIES: AtomicU64 = AtomicU64::new(0);
    #[inline]
    pub fn note_cofacet_scan(visited: u64) {
        COFACET_SCAN.fetch_add(visited, Ordering::Relaxed);
    }
    #[inline]
    pub fn note_cofacet_fill(visited: u64) {
        COFACET_FILL.fetch_add(visited, Ordering::Relaxed);
    }
    #[inline]
    pub fn note_cofacet_add(visited: u64) {
        COFACET_ADD.fetch_add(visited, Ordering::Relaxed);
    }
    #[inline]
    pub fn note_reduction_add(entries: u64) {
        REDUCTION_ADDS.fetch_add(1, Ordering::Relaxed);
        MATRIX_ENTRIES.fetch_add(entries, Ordering::Relaxed);
    }
    /// Pivots created by real reductions. Few (298,670 at dim=2), so they can
    /// be kept in a plain Vec without a lock from the single-threaded walk.
    pub static REDUCTION_PIVOTS: std::sync::Mutex<Vec<(crate::ripser::types::IndexT, usize)>> =
        std::sync::Mutex::new(Vec::new());

    pub fn note_reduction_pivot(key: crate::ripser::types::IndexT, pos: usize) {
        let mut g = REDUCTION_PIVOTS.lock().unwrap_or_else(|e| e.into_inner());
        g.push((key, pos));
    }

    pub fn report(
        dim: crate::ripser::types::IndexT,
        apparent: Option<&[crate::ripser::types::DiameterEntryT]>,
    ) {
        if std::env::var("CANNS_RIPSER_STATS").is_ok() {
            eprintln!(
                "WALKSTATS dim={} membership_queries={} hits={} \
                 heap_fills={} heap_entries={} scan_cols={} slow_cols={} \
                 cofacet_scan={} cofacet_fill={} cofacet_add={} \
                 reduction_adds={} matrix_entries={} lookup_s={:.3} \
                 cancels={} apparent_cols={} apparent_bars={}",
                dim,
                QUERIES.load(Ordering::Relaxed),
                HITS.load(Ordering::Relaxed),
                HEAP_FILLS.load(Ordering::Relaxed),
                HEAP_ENTRIES.load(Ordering::Relaxed),
                SCAN_COLUMNS.load(Ordering::Relaxed),
                SLOW_COLUMNS.load(Ordering::Relaxed),
                COFACET_SCAN.load(Ordering::Relaxed),
                COFACET_FILL.load(Ordering::Relaxed),
                COFACET_ADD.load(Ordering::Relaxed),
                REDUCTION_ADDS.load(Ordering::Relaxed),
                MATRIX_ENTRIES.load(Ordering::Relaxed),
                LOOKUP_NS.load(Ordering::Relaxed) as f64 / 1e9,
                CANCELLATIONS.load(Ordering::Relaxed),
                APPARENT_COLS.load(Ordering::Relaxed),
                APPARENT_BARS.load(Ordering::Relaxed)
            );
            if let Some(pre) = apparent {
                let g = REDUCTION_PIVOTS.lock().unwrap_or_else(|e| e.into_inner());
                let apparent_keys: std::collections::HashSet<i64> = pre
                    .iter()
                    .filter(|e| e.get_index() != -1)
                    .map(|e| e.get_index())
                    .collect();
                let mut collide = 0usize;
                let mut apparent_wins = 0usize;
                for (k, pos) in g.iter() {
                    if apparent_keys.contains(k) {
                        collide += 1;
                        if let Some(e) = pre.iter().find(|e| e.get_index() == *k) {
                            let _ = e;
                        }
                    }
                    let _ = pos;
                }
                // Which owner comes first in column order?
                let mut red_before = 0usize;
                let mut red_after = 0usize;
                let ap_pos: std::collections::HashMap<i64, usize> = pre
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.get_index() != -1)
                    .map(|(i, e)| (e.get_index(), i))
                    .collect();
                for (k, pos) in g.iter() {
                    if let Some(&ap) = ap_pos.get(k) {
                        if *pos < ap {
                            red_before += 1;
                        } else {
                            red_after += 1;
                        }
                    }
                }
                eprintln!(
                    "WALKCROSS dim={} reduction_pivots={} collide_with_apparent={} \
                     reduction_first={} apparent_first={}",
                    dim,
                    g.len(),
                    collide,
                    red_before,
                    red_after
                );
            }
        }
    }
}

#[cfg(not(feature = "apparent-stats"))]
mod walk_stats {
    #[inline(always)]
    pub fn count_query() {}
    #[inline(always)]
    pub fn count_hit() {}
    #[inline(always)]
    pub fn note_heap_fill(_e: usize) {}
    #[inline(always)]
    pub fn note_scan_column() {}
    #[inline(always)]
    pub fn note_slow_column() {}
    #[inline(always)]
    pub fn note_lookup_ns(_ns: u64) {}
    #[inline(always)]
    pub fn note_apparent_bar() {}
    #[inline(always)]
    pub fn note_apparent_col() {}
    #[inline(always)]
    pub fn note_cofacet_scan(_v: u64) {}
    #[inline(always)]
    pub fn note_cofacet_fill(_v: u64) {}
    #[inline(always)]
    pub fn note_cofacet_add(_v: u64) {}
    #[inline(always)]
    pub fn note_cancel() {}
    #[inline(always)]
    pub fn note_reduction_add(_e: u64) {}
    pub fn note_reduction_pivot(_k: crate::ripser::types::IndexT, _p: usize) {}
    pub fn report(
        _dim: crate::ripser::types::IndexT,
        _a: Option<&[crate::ripser::types::DiameterEntryT]>,
    ) {
    }
}

// Matrix reduction coordinator
pub struct MatrixReducer<M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + HasCofacets + Sync,
{
    pub dist: M,
    pub n: IndexT,
    pub threshold: ValueT,
    pub ratio: f32,
    pub modulus: CoefficientT,
    pub binomial_coeff: BinomialCoeffTable,
    pub multiplicative_inverse: Vec<CoefficientT>,
    pub verbose: bool,
}

impl<M> MatrixReducer<M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + HasCofacets + Sync,
{
    pub fn new(
        dist: M,
        n: IndexT,
        threshold: ValueT,
        ratio: f32,
        modulus: CoefficientT,
        binomial_coeff: BinomialCoeffTable,
        verbose: bool,
    ) -> Result<Self, String> {
        let multiplicative_inverse = multiplicative_inverse_vector(modulus)?;

        Ok(Self {
            dist,
            n,
            threshold,
            ratio,
            modulus,
            binomial_coeff,
            multiplicative_inverse,
            verbose,
        })
    }

    pub fn get_edge_vertices(&self, idx: IndexT, n: IndexT) -> (IndexT, IndexT) {
        // Ensure idx is within valid range
        debug_assert!(idx >= 0, "Edge index must be non-negative");
        debug_assert!(
            idx < self.binomial_coeff.get(n, 2),
            "Edge index out of bounds"
        );

        // Use safe binary search
        let mut i = n - 1;
        let mut step = i >> 1;

        while step > 0 {
            let mid = i - step;
            let binom_mid = self.binomial_coeff.get(mid, 2);
            if binom_mid > idx {
                i = mid;
            }
            step >>= 1;
        }

        // Final check
        while self.binomial_coeff.get(i, 2) > idx {
            i -= 1;
        }

        let j = idx - self.binomial_coeff.get(i, 2);
        (i, j)
    }

    pub fn get_simplex_vertices(&self, index: IndexT, dim: IndexT, n: IndexT) -> Vec<IndexT> {
        let mut vertices = Vec::with_capacity((dim + 1) as usize);
        let mut idx = index;
        let mut k = dim + 1;

        for i in (0..n).rev() {
            if k > 0 && idx >= self.binomial_coeff.get(i, k) {
                idx -= self.binomial_coeff.get(i, k);
                vertices.push(i);
                k -= 1;
            }
            if k == 0 {
                break;
            }
        }

        vertices.reverse();
        vertices
    }

    /// The "zero pivot cofacet" of `simplex`: the first cofacet (youngest =
    /// largest index, since the coboundary enumerator descends by index) whose
    /// diameter equals the simplex diameter. Sentinel (index -1) if none.
    fn get_zero_pivot_cofacet(&self, simplex: DiameterEntryT, dim: IndexT) -> DiameterEntryT {
        let target = simplex.get_diameter();
        let mut cofacets = self.dist.cofacet_enumerator(
            simplex,
            dim,
            self.n,
            &self.binomial_coeff,
            self.modulus,
            self.threshold,
        );
        while cofacets.has_next(true) {
            let cofacet = cofacets.next();
            apparent_stats::count_cofacet();
            if cofacet.get_diameter() > self.threshold {
                continue;
            }
            if cofacet.get_diameter() == target {
                return cofacet;
            }
        }
        DiameterEntryT::new(-1.0, -1, 0)
    }

    /// Decode simplex vertices (ascending) into a stack buffer, returning count.
    /// Avoids the Vec allocation of `get_simplex_vertices` on the hot apparent
    /// path.
    #[inline]
    fn fill_vertices(&self, index: IndexT, dim: IndexT, out: &mut [IndexT]) -> usize {
        let count = (dim + 1) as usize;
        let mut idx = index;
        let mut k = dim + 1;
        let mut pos = count;
        // Unrank the combinatorial number system by BINARY SEARCH.
        //
        // The previous formulation walked `i` from n-1 down to 0 and greedily
        // took the first i with C(i,k) <= idx. That is O(n) per call: at n=1200
        // the average simplex has a small minimum vertex, so the walk covers
        // ~900 indices before the rank is exhausted -- 900 binomial-table loads
        // to recover 3 or 4 vertices. With a fixed cofacets-per-column of 2.51
        // (measured) this decode, not the distance loads, dominates the
        // apparent pre-pass.
        //
        // C(i,k) is monotonically increasing in i, so the same vertex can be
        // found in O(log n). Same answer, same order (largest first, written
        // from the back so `out` ends up ascending); ~44 steps instead of ~900.
        // `hi` is exclusive -- the largest index still allowed at this rank.
        let mut hi = self.n;
        while k > 0 {
            // C(k-1, k) == 0 <= idx holds for every valid index, so `lo` is
            // always a legitimate lower bound rather than a sentinel.
            let mut lo = k - 1;
            let mut top = hi;
            while lo + 1 < top {
                let mid = lo + ((top - lo) >> 1);
                if self.binomial_coeff.get(mid, k) <= idx {
                    lo = mid;
                } else {
                    top = mid;
                }
            }
            out[pos - 1] = lo; // fill from the back → ascending order
            pos -= 1;
            idx -= self.binomial_coeff.get(lo, k);
            hi = lo;
            k -= 1;
        }
        count
    }

    /// The "zero pivot facet" of `simplex`: among its facets whose diameter
    /// equals the simplex diameter, the oldest one (smallest combinatorial
    /// index). Sentinel (index -1) if none. Allocation-free.
    ///
    /// NEGATIVE RESULT, kept so it is not retried: the loop below reads every
    /// edge nv-2 times (a tetrahedron costs 4x3 = 12 loads where only 6 edges
    /// exist), which looks like free savings. It is not. The duplicate loads
    /// hit L1 -- only 6 distinct cache lines are ever touched -- while hoisting
    /// them into a shared `facet_diam[]` costs 24 conditional `.max()` updates
    /// per call. Measured on the ASA matrix (3 interleaved passes, pre-pass
    /// segment only): 33.1 s -> 38.2 s, a reproducible ~16% REGRESSION.
    /// Redundancy in instructions is not redundancy in memory traffic.
    fn get_zero_pivot_facet(&self, simplex: DiameterEntryT, dim: IndexT) -> DiameterEntryT {
        const MAX_V: usize = 16;
        let target = simplex.get_diameter();
        let mut verts = [0 as IndexT; MAX_V];
        let nv = self.fill_vertices(simplex.get_index(), dim, &mut verts);
        let mut best_index: IndexT = -1;
        for remove in 0..nv {
            // facet = simplex minus verts[remove]; compute its diameter (max
            // pairwise distance among remaining vertices) and combinatorial index.
            let mut facet_diam: ValueT = 0.0;
            for a in 0..nv {
                if a == remove {
                    continue;
                }
                for b in (a + 1)..nv {
                    if b == remove {
                        continue;
                    }
                    facet_diam =
                        facet_diam.max(self.dist.get(verts[a] as usize, verts[b] as usize));
                }
            }
            if facet_diam == target {
                let mut facet_index: IndexT = 0;
                let mut rank = 1;
                for (a, &v) in verts.iter().take(nv).enumerate() {
                    if a == remove {
                        continue;
                    }
                    facet_index += self.binomial_coeff.get(v, rank);
                    rank += 1;
                }
                if best_index == -1 || facet_index < best_index {
                    best_index = facet_index;
                }
            }
        }
        DiameterEntryT::new(target, best_index, 1)
    }

    /// If `simplex` is the birth (facet) side of a zero-persistence apparent
    /// pair, return the paired cofacet; otherwise sentinel (index -1). The pair
    /// is apparent iff simplex's zero-pivot-cofacet exists and, reciprocally,
    /// that cofacet's zero-pivot-facet is `simplex`.
    pub fn get_zero_apparent_cofacet(
        &self,
        simplex: DiameterEntryT,
        dim: IndexT,
    ) -> DiameterEntryT {
        let cofacet = self.get_zero_pivot_cofacet(simplex, dim);
        if cofacet.get_index() == -1 {
            return DiameterEntryT::new(-1.0, -1, 0);
        }
        let facet = self.get_zero_pivot_facet(cofacet, dim + 1);
        if facet.get_index() == simplex.get_index() {
            cofacet
        } else {
            DiameterEntryT::new(-1.0, -1, 0)
        }
    }

    /// The P1/P2 seam: apparent-pair detection, hoisted out of the sequential
    /// walk so it can be evaluated ahead of time (and, in P2, in parallel).
    ///
    /// Returns the paired cofacet, or the (-1, -1, 0) sentinel when `entry` is
    /// not the birth side of a zero-apparent pair.
    ///
    /// This is deliberately a thin wrapper over `get_zero_apparent_cofacet`: the
    /// value of the seam is that it names the boundary, not that it changes the
    /// algorithm. It reads only `self.dist`, `self.binomial_coeff`, `self.n`
    /// and `self.threshold` -- never `pivot_column_index` -- which is what makes
    /// running it concurrently safe without any lock.
    #[inline]
    pub fn detect_apparent(&self, entry: DiameterEntryT, dim: IndexT) -> DiameterEntryT {
        apparent_stats::count_column();
        self.get_zero_apparent_cofacet(entry, dim)
    }

    pub fn normalize_coefficient(&self, coeff: CoefficientT) -> CoefficientT {
        // Normalize coefficient in Z/pZ field (ensure it's in range [0, p-1])
        let mut result = coeff % self.modulus;
        if result < 0 {
            result += self.modulus;
        }
        result
    }

    #[inline(always)]
    pub fn pop_pivot(&self, column: &mut WorkingT) -> DiameterEntryT {
        let mut pivot = DiameterEntryT::new(-1.0, -1, 0);

        while let Some(entry) = column.pop() {
            if pivot.get_coefficient() == 0 {
                pivot = entry;
            } else if entry.get_index() != pivot.get_index() {
                column.push(entry);
                return pivot;
            } else {
                // `modp(a + b, p)` where both coefficients are already in
                // [0, p): the sum is in [0, 2p), so the reduction is ONE
                // conditional subtraction and the integer division (an `idiv`,
                // ~25 cycles on x86) is pure waste. This is the innermost
                // operation of the reduction: it runs once per coboundary
                // cancellation, and the dim-1 reduction does tens of
                // millions of them.
                //
                // The sum is computed in i32 rather than i16: the old i16
                // addition overflowed for a modulus above 16383.
                let m = self.modulus as i32;
                let s = pivot.get_coefficient() as i32 + entry.get_coefficient() as i32;
                let new_coeff = (if s >= m { s - m } else { s }) as CoefficientT;
                pivot.set_coefficient(new_coeff);
                walk_stats::note_cancel();
            }
        }

        if pivot.get_coefficient() == 0 {
            DiameterEntryT::new(-1.0, -1, 0)
        } else {
            pivot
        }
    }

    #[inline(always)]
    pub fn get_pivot(&self, column: &mut WorkingT) -> DiameterEntryT {
        let result = self.pop_pivot(column);
        if result.get_index() != -1 {
            column.push(result);
        }
        result
    }

    // Superseded by scan_coboundary_pivot + fill_coboundary_heap on the hot
    // path; retained as the reference single-pass formulation.
    #[allow(dead_code)]
    pub fn init_coboundary_and_get_pivot(
        &self,
        simplex: DiameterEntryT,
        working_coboundary: &mut WorkingT,
        dim: IndexT,
        pivot_column_index: &PivotLookup<'_>,
        before: usize,
    ) -> DiameterEntryT {
        let mut check_for_emergent_pair = true;

        let mut cofacets = self.dist.cofacet_enumerator(
            simplex,
            dim,
            self.n,
            &self.binomial_coeff,
            self.modulus,
            self.threshold,
        );

        while cofacets.has_next(true) {
            let cofacet = cofacets.next();
            if cofacet.get_diameter() <= self.threshold {
                // first check for emergent pair
                if check_for_emergent_pair && simplex.get_diameter() == cofacet.get_diameter() {
                    if !pivot_column_index.contains(cofacet.get_index(), before) {
                        return cofacet;
                    }
                    check_for_emergent_pair = false;
                }
                working_coboundary.push(cofacet);
            }
        }

        self.get_pivot(working_coboundary)
    }

    /// Find the coboundary pivot of `simplex` via a single linear scan, without
    /// building the working-coboundary heap. The initial coboundary of a single
    /// simplex has all-distinct cofacet indices (coefficient 1), so no
    /// cancellation is possible and the pivot is simply the "maximal" cofacet
    /// (smallest diameter, largest index) under the `DiameterEntryT` ordering.
    ///
    /// Returns the emergent cofacet immediately if one is found (matching
    /// `init_coboundary_and_get_pivot`), or a sentinel (index -1) if the
    /// coboundary is empty.
    ///
    /// `buf` collects the cofacets this scan visits, in the same order and
    /// under the same threshold filter as `fill_coboundary_heap` would. That
    /// lets the slow path build the working heap from the scan's own results
    /// instead of enumerating the whole coboundary a SECOND time -- which is
    /// what ripser++ does with its `cofacet_entries` vector. The two enumerations
    /// are identical, so the heap content, and therefore the barcode, is
    /// unchanged.
    pub fn scan_coboundary_pivot(
        &self,
        simplex: DiameterEntryT,
        dim: IndexT,
        pivot_column_index: &PivotLookup<'_>,
        before: usize,
        buf: &mut Vec<DiameterEntryT>,
    ) -> DiameterEntryT {
        let mut check_for_emergent_pair = true;
        let mut pivot = DiameterEntryT::new(-1.0, -1, 0);
        let mut visited = 0u64;
        buf.clear();

        let mut cofacets = self.dist.cofacet_enumerator(
            simplex,
            dim,
            self.n,
            &self.binomial_coeff,
            self.modulus,
            self.threshold,
        );

        while cofacets.has_next(true) {
            let cofacet = cofacets.next();
            if cofacet.get_diameter() <= self.threshold {
                visited += 1;
                if check_for_emergent_pair && simplex.get_diameter() == cofacet.get_diameter() {
                    if !pivot_column_index.contains(cofacet.get_index(), before) {
                        walk_stats::note_cofacet_scan(visited);
                        return cofacet;
                    }
                    check_for_emergent_pair = false;
                }
                buf.push(cofacet);
                // Track the maximal cofacet (the heap pivot). Guard the sentinel
                // explicitly: its negative diameter would misorder under the
                // bit-pattern comparison used by the fast Ord.
                if pivot.get_index() == -1 || cofacet.cmp(&pivot) == std::cmp::Ordering::Greater {
                    pivot = cofacet;
                }
            }
        }
        walk_stats::note_cofacet_scan(visited);

        pivot
    }

    /// Populate `working_coboundary` with the full coboundary of `simplex`
    /// (cofacets under threshold), without touching the reduction column. Used
    /// on the slow path once a pivot collision is known to require reduction.
    pub fn fill_coboundary_heap(
        &self,
        simplex: DiameterEntryT,
        dim: IndexT,
        working_coboundary: &mut WorkingT,
    ) {
        let mut visited = 0u64;
        let mut cofacets = self.dist.cofacet_enumerator(
            simplex,
            dim,
            self.n,
            &self.binomial_coeff,
            self.modulus,
            self.threshold,
        );

        while cofacets.has_next(true) {
            let cofacet = cofacets.next();
            if cofacet.get_diameter() <= self.threshold {
                visited += 1;
                working_coboundary.push(cofacet);
            }
        }
        walk_stats::note_cofacet_fill(visited);
    }

    pub fn add_simplex_coboundary(
        &self,
        simplex: DiameterEntryT,
        dim: IndexT,
        working_reduction_column: &mut WorkingT,
        working_coboundary: &mut WorkingT,
    ) {
        working_reduction_column.push(simplex);
        let mut visited = 0u64;
        let mut cofacets = self.dist.cofacet_enumerator(
            simplex,
            dim,
            self.n,
            &self.binomial_coeff,
            self.modulus,
            self.threshold,
        );

        while cofacets.has_next(true) {
            let cofacet = cofacets.next();
            if cofacet.get_diameter() <= self.threshold {
                visited += 1;
                working_coboundary.push(cofacet);
            }
        }
        walk_stats::note_cofacet_add(visited);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_coboundary(
        &self,
        reduction_matrix: &CompressedSparseMatrix<DiameterEntryT>,
        columns_to_reduce: &[DiameterIndexT],
        index_column_to_add: usize,
        factor: CoefficientT,
        dim: IndexT,
        working_reduction_column: &mut WorkingT,
        working_coboundary: &mut WorkingT,
    ) {
        let column_to_add = DiameterEntryT::new(
            columns_to_reduce[index_column_to_add].get_diameter(),
            columns_to_reduce[index_column_to_add].get_index(),
            factor,
        );

        self.add_simplex_coboundary(
            column_to_add,
            dim,
            working_reduction_column,
            working_coboundary,
        );

        for simplex in reduction_matrix.subrange(index_column_to_add) {
            let mut modified_simplex = *simplex;
            let prod = (modified_simplex.get_coefficient() as i32) * (factor as i32);
            let new_coeff = modp(prod as CoefficientT, self.modulus);
            modified_simplex.set_coefficient(new_coeff);

            self.add_simplex_coboundary(
                modified_simplex,
                dim,
                working_reduction_column,
                working_coboundary,
            );
        }
    }

    // SoA version of add_coboundary for better cache performance
    #[allow(clippy::too_many_arguments)]
    pub fn add_coboundary_soa(
        &self,
        reduction_matrix_soa: &OptimizedSparseMatrix,
        columns_to_reduce: &[DiameterIndexT],
        index_column_to_add: usize,
        factor: CoefficientT,
        dim: IndexT,
        working_reduction_column: &mut WorkingT,
        working_coboundary: &mut WorkingT,
    ) {
        let column_to_add = DiameterEntryT::new(
            columns_to_reduce[index_column_to_add].get_diameter(),
            columns_to_reduce[index_column_to_add].get_index(),
            factor,
        );

        // NEGATIVE RESULT, measured: the pivot's OWN coboundary cannot be
        // dropped, even though ripser++'s `add_coboundary_reduction_submatrix`
        // appears to add only the submatrix entries' coboundaries and never
        // re-enumerates the pivot's own.
        //
        // The counters say the "extra" half is the big one: at dim 2 this
        // call re-enumerates and re-pushes the pivot's coboundary 434,515
        // times, i.e. 434,515 x 1197 = 5.2e8 cofacet enumerations and heap
        // pushes, while the reduction matrix itself is nearly empty
        // (17,842 entries). ripser++'s dim-2 reduction is 13.1 s against our
        // 14.8 s, so ~5e8 extra pushes cannot be the whole story -- and
        // dropping them does not even produce a WRONG barcode: the run
        // HANGS (0% CPU, idle GPU). Whatever ripser++ does differently, it
        // is not "skip the pivot's own coboundary".
        //
        // Tested behind `CANNS_RIPSER_SUBMATRIX_ONLY=1`, which hung, and the
        // switch was removed again. Do not retry it without first
        // understanding what ripser++'s working column actually contains.
        self.add_simplex_coboundary(
            column_to_add,
            dim,
            working_reduction_column,
            working_coboundary,
        );

        // Process SoA matrix with better cache locality
        let (diameters, indices, coefficients) = reduction_matrix_soa.subrange(index_column_to_add);
        walk_stats::note_reduction_add(diameters.len() as u64);

        for i in 0..diameters.len() {
            let prod = (coefficients[i] as i32) * (factor as i32);
            let new_coeff = modp(prod as CoefficientT, self.modulus);

            let modified_simplex = DiameterEntryT::new(diameters[i], indices[i], new_coeff);

            self.add_simplex_coboundary(
                modified_simplex,
                dim,
                working_reduction_column,
                working_coboundary,
            );
        }
    }

    pub fn compute_pairs(
        &self,
        columns_to_reduce: &[DiameterIndexT],
        pivot_column_index: &mut FxHashMap<IndexT, PivotVal>,
        dim: IndexT,
        progress_callback: &Option<pyo3::Py<pyo3::PyAny>>,
        progress_update_interval: std::time::Duration,
        last_progress_update: &mut Option<std::time::Instant>,
        births_and_deaths: &mut Vec<ValueT>,
        cocycles: &mut Vec<Vec<i32>>,
        do_cocycles: bool,
        // True when the NEXT dimension's assembly will query this map for
        // membership. Assembly bypasses `PivotLookup` and reads this map
        // directly (`keys()` for the GPU pivot list, `contains_key()` in the
        // CPU fallback), so while it is true the apparent entries MUST stay in
        // the hash. When false they may live in `ShardedPivotTable` instead,
        // which is what lets dim=2 at maxdim=2 skip 286.5M sequential inserts.
        pivot_map_needed_for_assembly: bool,
    ) {
        // Computing pairs for dimension
        if self.verbose {
            eprintln!(
                "DEBUG: compute_pairs dim={}, processing {} columns",
                dim,
                columns_to_reduce.len()
            );
        }

        // Initialize progress reporting
        let should_report_progress = progress_callback.is_some() && !columns_to_reduce.is_empty();
        if should_report_progress {
            if self.verbose {
                eprintln!(
                    "DEBUG: Setting up progress reporting for {} columns in dimension {}",
                    columns_to_reduce.len(),
                    dim
                );
            }
            // Report progress start to Python callback if available
            if let Some(ref callback) = progress_callback {
                pyo3::Python::attach(|py| {
                    let _ = callback.call1(
                        py,
                        (
                            0,
                            columns_to_reduce.len(),
                            format!("Computing H{} pairs", dim),
                        ),
                    );
                });
            }
        }

        // Use SoA (Structure of Arrays) layout for better cache performance
        let mut reduction_matrix_soa = OptimizedSparseMatrix::new();
        // Initial capacity is only a hint: preserve all entries via normal Vec growth.
        // Avoid multi-GiB virtual reservations for mostly trivial reduction columns.
        let initial_entries = columns_to_reduce.len().saturating_mul(10).min(1 << 20);
        reduction_matrix_soa.reserve(initial_entries);
        // The bounds array holds one entry per NON-EMPTY column (298,670 at
        // dim=2 on the ASA matrix), not one per enumerated column, so it needs
        // no up-front reservation: the old `reserve_columns(columns_to_reduce.len())`
        // reserved 1.15 GB of u32 for a structure that holds 2.4 MB.

        // Pre-allocate working buffers with estimated capacity based on problem size
        let estimated_working_size = std::cmp::min(1000, columns_to_reduce.len());
        let mut working_reduction_column = WorkingT::with_capacity(estimated_working_size);
        let mut working_coboundary = WorkingT::with_capacity(estimated_working_size);
        // The cofacets the reduction's scan visits, kept so the slow path can
        // build the working heap from them instead of enumerating the
        // coboundary again. Sized for the worst case (a dim-2 column's
        // coboundary is at most n-3 tetrahedra) and reused for every column,
        // so this is one allocation for the whole dimension.
        let mut cofacet_buf: Vec<DiameterEntryT> = Vec::with_capacity(estimated_working_size);
        // ONE-PASS COFACET ENUMERATION.
        //
        // The reduction used to walk each reducing column's coboundary
        // TWICE: once in `scan_coboundary_pivot`, and again in
        // `fill_coboundary_heap` when the scan found a pivot collision and
        // the real reduction had to happen. ripser++ walks it once -- it
        // keeps the cofacets in a `cofacet_entries` vector and builds the
        // heap from that.
        //
        // Kill switch: CANNS_RIPSER_ONE_PASS_COFACET=0 restores the two-pass
        // behaviour, so the change can be A/B-ed inside ONE batch instead of
        // across batches (the machine is shared and its load moves baselines
        // by 14% between batches).
        let one_pass_cofacet = env_forced("CANNS_RIPSER_ONE_PASS_COFACET").unwrap_or(true);
        let modulus = self.modulus; // local copy for closure

        // Apparent-pairs optimization: a column that is the birth side of a
        // zero-apparent pair is paired via a facet check, skipping reduction.
        //
        // The old default was OFF, justified by a measurement ("maxdim=2
        // ~1.16x -> ~0.67x") taken before P1-P14 existed, when the pre-pass was
        // a sequential scan. That measurement is stale and the decision was
        // wrong for the workload this library actually runs: re-measured over
        // the whole 112-cell grid with everything else in place,
        //
        //     cols >= 2^18 : 26/26 cells FASTER, median 1.18x, rising to 2.08x
        //                     at ~1e6+ columns
        //     cols <  2^17 : a wash -- median 1.01-1.05x, worst cell 0.92x,
        //                     and every loss is on a problem whose whole runtime
        //                     is single-digit milliseconds
        //
        // and on the ASA matrix at 1200^2 the effect is 97.97 s -> 50.69 s
        // (1.93x) on the same harness, same machine, same day. The benefit is
        // monotone in column count, so the gate is a column-count threshold
        // rather than a global on/off. Barcodes are unaffected either way: the
        // grid's `equiv` mode requires apparent-off and apparent-on results to
        // be bitwise identical and passes 224/224.
        //
        // CANNS_RIPSER_APPARENT still overrides in BOTH directions -- "0"/"off"
        // forces the old path even above the threshold, "1"/"on" forces it on
        // even below it, which is what the benchmark harness and the
        // A/B arms rely on.
        let apparent_enabled = !do_cocycles && apparent_requested(columns_to_reduce.len());
        let stats = std::env::var("CANNS_RIPSER_STATS").is_ok();
        let mut stat_apparent: u64 = 0;
        let mut stat_pre_pass: f64 = 0.0;
        let compute_t0 = std::time::Instant::now();

        // P1/P2: apparent-pair detection is a PURE function of
        // (column, dim) -- `get_zero_apparent_cofacet` reads only self.dist,
        // self.binomial_coeff, self.n and self.threshold, and never touches
        // pivot_column_index. So it can be hoisted out of the sequential walk
        // and evaluated ahead of time, in parallel, WITHOUT changing the order
        // in which pivots are discovered and recorded.
        //
        // The results are consumed by a chunk, never all at once: at maxdim=2
        // n=1200 there are C(1200,3) ~= 2.9e8 columns, and one DiameterEntryT
        // per column would be ~4.6 GB if materialised whole.
        let apparent_chunk = apparent_chunk_size();
        let mut apparent_pre: Vec<DiameterEntryT> = Vec::new();

        // P2: run the pre-pass across rayon threads when there is more than one.
        //
        // Safety argument: `detect_apparent` reads only &self (immutable) and
        // returns a value. It touches no shared mutable state -- no
        // pivot_column_index, no births_and_deaths, no SoA matrix. Those are
        // all written exclusively in the SEQUENTIAL walk below. So there is
        // nothing to lock, and no compare_exchange race of the kind that made
        // lockfree.rs 123x slower than sequential.
        //
        // With a single thread rayon is pure overhead, so take the plain loop.
        #[cfg(feature = "parallel")]
        let apparent_parallel =
            apparent_parallel_enabled() && rayon::current_num_threads() > 1 && apparent_chunk > 1;
        #[cfg(not(feature = "parallel"))]
        let apparent_parallel = false;

        // GPU path for the apparent pre-pass. Detection is a pure function of
        // (column, dim), so it can be done for every column at once instead of
        // chunk by chunk -- but only when the device exists and the batch is
        // big enough to pay for the readback. Everything below then reads
        // `gpu_pre` indexed by ABSOLUTE column number, while the CPU path reads
        // `apparent_pre` indexed by offset within the chunk. Both feed the same
        // consumption site, so the sequential walk is unchanged either way.
        //
        // `stat_gpu_s` is reported next to `stat_pre_pass` so the two can be
        // compared in one run rather than across sessions.
        // The GPU apparent population, read IN PLACE from the page-locked
        // buffer the detect kernel wrote. It used to be a host `Vec` built by
        // a parallel conversion pass, which cost 1.46 s and 4.6 GB at ASA
        // scale for a copy nobody needed: the kernel's 16-byte records have
        // the same layout as `DiameterEntryT`, so the buffer IS the vector.
        let mut gpu_pre = crate::ripser::gpu::ApparentRecords::empty();
        let mut use_gpu_pre = false;
        let mut stat_gpu_s = 0.0f64;
        let mut stat_densify = 0.0f64;
        let mut gpu_err = crate::ripser::gpu::last_error();
        if apparent_enabled && columns_to_reduce.len() >= crate::ripser::gpu::GPU_MIN_COLUMNS {
            let gpu_t0 = std::time::Instant::now();
            // `densify` unpacks the packed lower triangle into a dense n*n
            // matrix. It is INSIDE `gpu_pre_s` but OUTSIDE the GPUPHASE
            // timers, so whatever it costs used to land in the phase
            // breakdown as "unaccounted" -- and it was blamed on the
            // host-side materialisation, which turned out to be worth much
            // less than the phase gap suggested. Time it separately.
            let t_densify = std::time::Instant::now();
            let dense = crate::ripser::gpu::densify(&self.dist, self.n);
            stat_densify = t_densify.elapsed().as_secs_f64();
            match crate::ripser::gpu::detect(
                &dense,
                self.n,
                &self.binomial_coeff,
                columns_to_reduce,
                dim,
                self.threshold,
                self.modulus,
                self.ratio,
            ) {
                Ok(results) => {
                    // detect() returns the page-locked buffer itself; the
                    // records are read in place, so there is no host-side
                    // copy of the 4.6 GB population.
                    gpu_pre = results;
                    use_gpu_pre = true;
                }
                Err(e) => {
                    // Any device problem falls back silently: the CPU path is
                    // correct, just slower. The reason is reported under STATS
                    // because a silent fallback is otherwise indistinguishable
                    // from the GPU simply not being used.
                    gpu_err = e.clone();
                    if self.verbose {
                        eprintln!("DEBUG: GPU apparent pre-pass unavailable ({e}); using CPU");
                    }
                }
            }
            stat_gpu_s = gpu_t0.elapsed().as_secs_f64();
        }

        // NEGATIVE RESULT, reverted: the apparent pivot population was moved
        // out of the hash map into a bulk-built SORTED index here, and it
        // measured 59.1 s -> 84.6 s over interleaved A/B: 25 s SLOWER.
        //
        // Two costs, both measured rather than assumed:
        //
        //  * Building in situ cost 19.4 s, not the 6.3 s the isolated
        //    benchmark promised, because it walks the 4.6 GB `gpu_pre`
        //    instead of generating keys.
        //  * Lookups got far worse, not better. The real cofacet keys are
        //    heavily clustered -- consecutive present keys differ by a median
        //    of 1, and 82% are within 1e6 -- so the sorted index's bucket scan
        //    degenerated to 18.7 us per probe (3.9e6 probes = 73 s).
        //
        // Note what did NOT change: re-running the isolated benchmark with the
        // REAL keys gave a hash build of 25.9 s against 27.0 s for synthetic
        // ones. The key distribution is irrelevant to the hash cost here, so
        // "the synthetic keys flattered the hash table" is NOT the explanation.
        //
        // Conclusion: for this ACCESS PATTERN the sorted structure is wrong.
        // It said nothing about the BUILD cost, which is the half that
        // actually dominated. `ShardedPivotTable` keeps the hash access
        // pattern and parallelises only the construction.
        //
        // Only the GPU pre-pass materialises every column at once, so only it
        // can be bulk-built; the chunked CPU pre-pass leaves the sequential
        // inserts in place. On this workload the GPU path is always taken
        // (`GPU_MIN_COLUMNS` is far below C(1200,3)), but the CPU path must
        // keep working, so `use_gpu_pre` gates it.
        //
        // DO NOT DROP `!pivot_map_needed_for_assembly` FROM THIS CONDITION.
        // Assembly of the NEXT dimension reads this map directly and bypasses
        // `PivotLookup` entirely -- `assembly.rs` does
        // `pivot_column_index.keys()` for the GPU pivot list and
        // `.contains_key()` in the CPU fallback. If the apparent entries live
        // only in the sharded table, assembly of dim d+1 sees just the ~299K
        // reduction pivots and produces the wrong columns.
        //
        // That is not hypothetical: at maxdim=2, dim=1 has C(1200,2) = 719,400
        // columns, which clears `GPU_MIN_COLUMNS`, so dim=1 DOES get a sharded
        // table -- and dim=2's assembly consumes dim=1's map. Hence
        // `pivot_map_needed_for_assembly` (`dim < dim_max`), not "the last
        // dimension", is the correct condition. At maxdim=2 the win lands on
        // dim=2 (286.5M apparent inserts); at maxdim>=3 it would land on dim=3
        // instead, because dim=2's map has to stay a real hash.
        //
        // Removing this gate is the follow-up worth doing: give assembly a
        // bitmap over the simplex-index key space (C(1200,2) = 719,400 bits =
        // 90 KB; C(1200,3) = 2.9e8 bits = 36 MB) instead of a sorted pivot
        // list. Both fit in device memory, membership becomes one bit test, and
        // `pivot_map_needed_for_assembly` stops being a constraint at all.
        let sharded_pivot = sharded_pivot_requested(columns_to_reduce.len());
        // 3a: the CPU path can build the table too. It previously could not,
        // because the table was only ever filled from the GPU pre-pass
        // (`build_from_columns(&gpu_pre)`) while the CPU computes apparent
        // results chunk by chunk inside the walk. That made the CPU path
        // insert EVERY apparent column into the FxHashMap one at a time
        // -- 286,263,529 inserts at ~85 ns = the 24.4 s that `nonapparent`
        // instrumentation attributes to the apparent fast path. The GPU path
        // does zero such inserts because the table is already bulk-built.
        //
        // `detect_apparent` is a pure function of (column, dim): it reads only
        // dist / binomial_coeff / n / threshold and never touches the pivot
        // map, which is exactly what makes running it concurrently safe. So the
        // CPU can run ONE parallel pass over the whole dimension up front,
        // keep the result in `apparent_pre` (the walk needs the death values
        // anyway to emit `births_and_deaths`), and build the table from it --
        // no detection is done twice, and the per-chunk re-detection inside the
        // walk is skipped once the table exists.
        //
        // Kill switch: CANNS_RIPSER_CPU_SHARDED=0 restores the old path.
        let cpu_sharded = !use_gpu_pre && env_forced("CANNS_RIPSER_CPU_SHARDED").unwrap_or(true);
        let mut stat_prepass_hoisted = 0.0f64;
        let mut stat_sharded_build = 0.0f64;
        // PHASE1 SPLIT: time spent on columns that actually reduce, excluding
        // the apparent fast path and excluding `pre_pass`. See the timer site
        // below; the point is to make "what would 2b actually save?" a number.
        let mut stat_nonapparent_loop_s = 0.0f64;
        let mut stat_nonapparent_cols: u64 = 0;
        // The split of `stat_nonapparent_loop_s` into its three parts. All
        // three clocks are read ONCE PER REDUCING COLUMN, not once per
        // cofacet, so they cost ~25 ns x 3 x (number of reducing columns)
        // -- about 4 ms at ASA scale -- and do not distort what they
        // measure. (The `apparent-stats` counters are the opposite: they sit
        // in the innermost loop and inflate it threefold, which is why they
        // live in a separate cargo feature.)
        //
        //   scan_s   : scan_coboundary_pivot (the cofacet enumeration) plus
        //              the pivot-map probe that follows it.
        //   fill_s   : building the coboundary heap out of the scan's buffer,
        //              plus the first pop.
        //   reduce_s : the reduction rounds themselves -- each one re-enumerates
        //              the pivot's own coboundary and pushes it into the heap.
        let mut stat_scan_s = 0.0f64;
        let mut stat_fill_s = 0.0f64;
        let mut stat_reduce_s = 0.0f64;
        let mut stat_rounds: u64 = 0;
        // The peak size of the coboundary heap. It is checked once per
        // reduction round, so it costs nothing. The question it answers: at
        // dim 1 a column takes ~48 rounds and each round pushes its whole
        // 1198-cofacet coboundary, so the heap can grow to tens of thousands
        // of entries -- 1.5 MB -- and every sift then misses L1. At dim 2 a
        // column takes 0.9 rounds on average and the heap stays at ~600
        // entries. That would explain why a dim-1 round costs 4x a dim-2
        // round on BOTH implementations.
        let mut stat_heap_max: usize = 0;
        let mut stat_push_s = 0.0f64;
        let mut stat_pop_s = 0.0f64;
        let mut stat_redheap_max: usize = 0;
        let mut stat_col_rounds_max: u64 = 0;
        let mut stat_col_rounds_cur: u64 = 0;
        let apparent_table: Option<crate::ripser::pivot_sharded::ShardedPivotTable> =
            if sharded_pivot
                && apparent_enabled
                && !pivot_map_needed_for_assembly
                && (use_gpu_pre || cpu_sharded)
            {
                if !use_gpu_pre {
                    // One parallel pass over the whole dimension. The walk reads
                    // these back by absolute column index, so the per-chunk
                    // clear/reserve/extend inside the loop is skipped entirely
                    // once the table exists (guarded by `use_gpu_pre` below).
                    let t = std::time::Instant::now();
                    let n = columns_to_reduce.len();
                    apparent_pre.clear();
                    apparent_pre.reserve(n);
                    #[cfg(feature = "parallel")]
                    if apparent_parallel {
                        use rayon::prelude::*;
                        // rayon's Map is a ParallelIterator, NOT an Iterator,
                        // so it cannot feed Vec::extend directly -- collect
                        // first (same reason as the chunked pass below).
                        let detected: Vec<DiameterEntryT> = columns_to_reduce
                            .par_iter()
                            .map(|col| {
                                self.detect_apparent(
                                    DiameterEntryT::new(col.get_diameter(), col.get_index(), 1),
                                    dim,
                                )
                            })
                            .collect();
                        apparent_pre.extend(detected);
                    } else {
                        apparent_pre.extend(columns_to_reduce.iter().map(|col| {
                            self.detect_apparent(
                                DiameterEntryT::new(col.get_diameter(), col.get_index(), 1),
                                dim,
                            )
                        }));
                    }
                    #[cfg(not(feature = "parallel"))]
                    apparent_pre.extend(columns_to_reduce.iter().map(|col| {
                        self.detect_apparent(
                            DiameterEntryT::new(col.get_diameter(), col.get_index(), 1),
                            dim,
                        )
                    }));
                    stat_prepass_hoisted = t.elapsed().as_secs_f64();
                    stat_pre_pass = stat_prepass_hoisted;
                    if stats {
                        stat_apparent =
                            apparent_pre.iter().filter(|c| c.get_index() != -1).count() as u64;
                    }
                }
                let t = std::time::Instant::now();
                // The source differs by path, and picking the wrong one fails
                // SILENTLY: building from `apparent_pre` on the GPU path (where
                // nothing ever fills it) yields an EMPTY table -- no apparent
                // pivots, a wrong dim2 barcode, and a wall time that looks 2x
                // better because no apparent work happened at all. The unit
                // tests did not catch it; only the md5 gate did.
                let src = if use_gpu_pre {
                    gpu_pre.as_slice()
                } else {
                    apparent_pre.as_slice()
                };
                // NOT a debug_assert: benchmarks and the published wheel are
                // release builds, where debug assertions are compiled out --
                // which is exactly why the first version of this refactor
                // shipped a silently empty table on the GPU path and only the
                // md5 gate caught it. This runs once per dimension, so the
                // check is free where it matters.
                assert_eq!(
                    src.len(),
                    columns_to_reduce.len(),
                    "sharded pivot source must be column-aligned with \
                     columns_to_reduce (empty source => empty table => lost pivots)"
                );
                let built =
                    crate::ripser::pivot_sharded::ShardedPivotTable::build_from_columns(src);
                stat_sharded_build = t.elapsed().as_secs_f64();
                built
            } else {
                None
            };
        let _apparent_index: Option<&crate::ripser::pivot_sharded::ShardedPivotTable> =
            apparent_table.as_ref();

        // Reserve for what this map will actually hold, now that we know
        // whether the sharded table exists.
        //
        // With the table, apparent columns are NOT inserted here (the walk
        // guards on `_apparent_index.is_none()`), and WALKCROSS measured the
        // apparent and reduction key sets to be disjoint -- so the map can
        // only ever hold pivots from the remaining columns. At ASA scale that
        // is 298,670 entries instead of 286,562,199: reserving the full count
        // cost ~8.6 GB of hashbrown slots that were never written.
        //
        // Under-reserving is safe (hashbrown rehashes); over-reserving is
        // silent. So when in doubt this errs small, and the A/B gates on
        // barcode md5 plus wall time.
        let _reserve = match _apparent_index {
            Some(t) => columns_to_reduce
                .len()
                .saturating_sub(t.present_count())
                .max(1024),
            None => columns_to_reduce.len(),
        };
        let t_reserve = std::time::Instant::now();
        pivot_column_index.reserve(_reserve);
        let stat_reserve_s = t_reserve.elapsed().as_secs_f64();

        // STREAMREAD / WRITEPROBE -- see report S22.8 and S22.11.
        //
        // These probes are PURELY ADDITIVE: they allocate, fill and drop scratch
        // buffers, and compute a checksum. No algorithm state is read or
        // written, so the barcode is untouched and they cannot poison the pivot
        // map the way every ablation attempt did (S22.2, S22.3).
        //
        // They exist because the walk costs 18.1 s over 286,562,199 iterations
        // = 63 ns each, while moving only ~44 B per iteration -- an order of
        // magnitude under this machine's single-core bandwidth. Something is
        // stalling, and `perf` is unavailable here (perf_event_paranoid >= 1,
        // no root). S22.8 already showed the two READ streams together cost
        // 0.80 s, so the candidate list is down to the two WRITE patterns.
        #[cfg(feature = "walk-probe")]
        {
            let n = columns_to_reduce.len();

            let t = std::time::Instant::now();
            let mut a = 0f32;
            // `ApparentRecords` keeps the population in a pinned buffer; the
            // flat per-column view is `as_slice()`.
            for e in gpu_pre.as_slice().iter() {
                a += e.get_diameter();
            }
            let d_gpu = t.elapsed().as_secs_f64();

            let t = std::time::Instant::now();
            let mut b = 0f32;
            for e in columns_to_reduce.iter() {
                b += e.get_diameter();
            }
            let d_cols = t.elapsed().as_secs_f64();

            // `append_column`: one push into a Vec<u32> reserved up front.
            let t = std::time::Instant::now();
            let mut bounds: Vec<u32> = Vec::with_capacity(n);
            for i in 0..n {
                bounds.push(i as u32);
            }
            let d_bounds = t.elapsed().as_secs_f64();

            // `births_and_deaths`: two pushes per iteration into a Vec<f32>
            // that has to DOUBLE its way to 2*n elements.
            let t = std::time::Instant::now();
            let mut births: Vec<f32> = Vec::new();
            for i in 0..n {
                births.push(i as f32);
                births.push(i as f32);
            }
            let d_births = t.elapsed().as_secs_f64();

            eprintln!(
                "WALKPROBE cols={n} gpu_pre={d_gpu:.3} columns={d_cols:.3} \
                 bounds_push={d_bounds:.3} births_push={d_births:.3} \
                 sink={a:.3}{b:.3}{}{}",
                bounds.len(),
                births.len()
            );
            drop(bounds);
            drop(births);
        }

        // SPLIT: exactly TWO clock reads for the whole chunk loop, so the
        // distortion is nil (a per-iteration clock would cost 25 ns against a
        // 63 ns loop). This separates "the loop" from "everything else inside
        // compute_pairs", which until now has only ever been measured as one
        // number, `total_s`.
        let loop_t0 = std::time::Instant::now();
        // Everything between the start of compute_pairs and the start of the
        // chunk loop: the GPU pre-pass, the sharded pivot build, the pivot
        // map's reservation and the diagnostics. It was never a separate line
        // in the STATS block, so a cost there was invisible.
        let stat_pre_loop_s = loop_t0.duration_since(compute_t0).as_secs_f64();
        let pc_before = proc_counters();
        // Close the last unverified assumption: that the 18 s really is
        // 286M iterations of the apparent fast path. Counted, not assumed.
        let mut iters: u64 = 0;
        let mut apparent_iters: u64 = 0;

        for chunk_start in (0..columns_to_reduce.len()).step_by(apparent_chunk) {
            let chunk_end = std::cmp::min(chunk_start + apparent_chunk, columns_to_reduce.len());

            // Skipped when 3a already filled `apparent_pre` for the whole
            // dimension AND built the table from it: the per-chunk detection
            // would recompute every entry that is already resident.
            if apparent_enabled && !use_gpu_pre && apparent_table.is_none() {
                let pre_t0 = std::time::Instant::now();
                apparent_pre.clear();
                apparent_pre.reserve(chunk_end - chunk_start);
                let chunk = &columns_to_reduce[chunk_start..chunk_end];

                #[cfg(feature = "parallel")]
                if apparent_parallel {
                    // rayon's Map implements ParallelIterator, NOT Iterator, so
                    // it cannot feed Vec::extend directly -- collect first.
                    let detected: Vec<DiameterEntryT> = chunk
                        .par_iter()
                        .map(|col| {
                            let entry = DiameterEntryT::new(col.get_diameter(), col.get_index(), 1);
                            self.detect_apparent(entry, dim)
                        })
                        .collect();
                    apparent_pre.extend(detected);
                } else {
                    apparent_pre.extend(chunk.iter().map(|col| {
                        let entry = DiameterEntryT::new(col.get_diameter(), col.get_index(), 1);
                        self.detect_apparent(entry, dim)
                    }));
                }

                #[cfg(not(feature = "parallel"))]
                apparent_pre.extend(chunk.iter().map(|col| {
                    let entry = DiameterEntryT::new(col.get_diameter(), col.get_index(), 1);
                    self.detect_apparent(entry, dim)
                }));

                if stats {
                    stat_apparent +=
                        apparent_pre.iter().filter(|c| c.get_index() != -1).count() as u64;
                }
                // Wall time actually spent in the detection pre-pass. This is
                // the number the P3 (SIMD) gate is decided on: SIMD is only
                // worth writing if detection is a real share of the remaining
                // wall time AFTER parallelisation, not before.
                stat_pre_pass += pre_t0.elapsed().as_secs_f64();
            }

            for index_column_to_reduce in chunk_start..chunk_end {
                iters += 1;
                let column_to_reduce = &columns_to_reduce[index_column_to_reduce];
                // The modulo lives INSIDE the `verbose` arm, not in an `&&`
                // chain beside it. LLVM sinks the multiply for
                // `index % 1000` -- a 64-bit `imul` plus a rotate plus a
                // compare -- into the hot path and evaluates it on every one
                // of the 286,562,199 iterations before testing the
                // loop-invariant `verbose` flag. Confirmed by disassembly of
                // the release build:
                //     cmpb $0x0,0xec(%rsp)      ; self.verbose
                //     jne   <skip>             ; falls THROUGH to the modulo
                //     movabs $0xcccc...cccd,%rcx
                //     imul  %rcx,%rax
                //     ror   %rax
                //     movabs $0x1999...9999,%rcx
                //     cmp   %rcx,%rax
                // The nesting below is semantically identical and cannot be
                // sunk past the branch.
                if self.verbose {
                    if index_column_to_reduce % 1000 == 0 {
                        eprintln!(
                            "DEBUG: compute_pairs dim={}, column {}/{}",
                            dim,
                            index_column_to_reduce,
                            columns_to_reduce.len()
                        );
                    }
                }

                // Progress reporting. `Instant::now()` used to be evaluated for
                // EVERY column (286.6M calls at maxdim=2, ~25 ns each) even
                // though the callback only fires on every 10th. Gate the clock
                // read on the same modulus the callback uses: same callback
                // rate, one clock read per 10 columns instead of per column.
                let should_update_progress =
                    if should_report_progress && index_column_to_reduce % 10 == 0 {
                        let now = std::time::Instant::now();
                        match last_progress_update {
                            Some(last) => now.duration_since(*last) >= progress_update_interval,
                            None => true,
                        }
                    } else {
                        false
                    };

                if should_update_progress {
                    *last_progress_update = Some(std::time::Instant::now());
                    if let Some(ref callback) = progress_callback {
                        pyo3::Python::attach(|py| {
                            let _ = callback.call1(
                                py,
                                (
                                    index_column_to_reduce,
                                    columns_to_reduce.len(),
                                    format!("Computing H{} pairs", dim),
                                ),
                            );
                        });
                    }
                }

                // `begin_column` MUST stay before the apparent fast path:
                // `reduction_matrix_soa.subrange(index_column_to_add)` is indexed
                // by column number, so the store has to know which column it is
                // filling. Do not "optimise" this away.
                //
                // The store keeps one bound per NON-EMPTY column, so this call
                // does not allocate: a column that pushes no entry (an
                // apparent pair) never gets a bound at all.
                reduction_matrix_soa.begin_column(index_column_to_reduce);

                // THE GPU ALREADY KNOWS. The detect kernel computed, per
                // column, whether the column is an apparent pair AND whether
                // it contributes a bar (it has both the column's own diameter
                // and the partner's, so it can apply the ratio filter).
                //
                // Without this, the host had to read the 4.6 GB apparent
                // population AND the 4.6 GB of column diameters for all
                // 286,981,060 apparent columns -- to push bars that, at this
                // scale, were ZERO (`apparent_bars=0` in the WALKSTATS line).
                // That walk cost 1.55 s of the dim-2 reduction.
                //
                // The ORDER is unchanged: the same loop, the same columns in
                // the same order, the same pushes. Only the reads are gone.
                if use_gpu_pre && apparent_enabled && gpu_pre.is_apparent(index_column_to_reduce) {
                    apparent_iters += 1;
                    walk_stats::note_apparent_col();
                    // The bar. The GPU already applied the ratio filter (it
                    // has both the column's own diameter and the partner's),
                    // so the host does not have to read the 4.6 GB
                    // population to find out.
                    if gpu_pre.produces_bar(index_column_to_reduce) {
                        let apparent = gpu_pre.as_slice()[index_column_to_reduce];
                        let diameter = columns_to_reduce[index_column_to_reduce].get_diameter();
                        let death = apparent.get_diameter();
                        walk_stats::note_apparent_bar();
                        if death > diameter * self.ratio {
                            births_and_deaths.push(diameter);
                            births_and_deaths.push(death);
                        }
                    }
                    // ⚠️ THE PIVOT INSERT IS NOT OPTIONAL. When the NEXT
                    // dimension's assembly needs the pivot map, the sharded
                    // table is not built (`pivot_map_needed_for_assembly`),
                    // so `_apparent_index` is None and the apparent pivots
                    // have to be inserted into the FxHashMap one at a time.
                    //
                    // Dropping this made the reduction hang: the pivot map
                    // was missing entries, so the walk looked for a pivot
                    // that was never created. It only bites on the non-last
                    // dimension, which is why a last-dimension-only test
                    // would not have caught it.
                    if _apparent_index.is_none() {
                        let apparent = gpu_pre.as_slice()[index_column_to_reduce];
                        pivot_column_index.insert(
                            apparent.get_index(),
                            PivotVal::new(index_column_to_reduce, apparent.get_coefficient()),
                        );
                    }
                    continue;
                }

                let column_to_reduce_entry = DiameterEntryT::new(
                    column_to_reduce.get_diameter(),
                    column_to_reduce.get_index(),
                    1,
                );
                let diameter = column_to_reduce_entry.get_diameter();

                // Apparent-pair fast path: if this column is the birth side of a
                // zero-apparent pair, its death is known immediately (a cheap
                // facet check) with no coboundary heap or reduction at all.
                //
                // The detection already happened in the chunk pre-pass; this
                // block only CONSUMES its result. Everything written here
                // (births_and_deaths, pivot_column_index) stays strictly
                // sequential and in the original column order, which is what
                // keeps the barcode identical to the pre-refactor behaviour.
                //
                // Moved ABOVE the scratch-buffer reset on purpose. Those buffers
                // are dead the moment we `continue`, yet the old order reset
                // them (two clears, two capacity checks, a push) for all
                // 286.6M columns before discovering 99.89% of them were
                // apparent pairs. Resetting them is provably a no-op for those
                // columns, since the next iteration clears them again.
                if apparent_enabled {
                    let apparent = if use_gpu_pre {
                        gpu_pre.as_slice()[index_column_to_reduce]
                    } else {
                        // Full-length when 3a hoisted the pass; chunk-offset otherwise.
                        if apparent_pre.len() == columns_to_reduce.len() {
                            apparent_pre[index_column_to_reduce]
                        } else {
                            apparent_pre[index_column_to_reduce - chunk_start]
                        }
                    };
                    if apparent.get_index() != -1 {
                        apparent_iters += 1;
                        walk_stats::note_apparent_col();
                        let death = apparent.get_diameter();
                        #[cfg(not(feature = "ablate-birth-push"))]
                        if death > diameter * self.ratio {
                            walk_stats::note_apparent_bar();
                            births_and_deaths.push(diameter);
                            births_and_deaths.push(death);
                        }
                        // ATTRIBUTION ONLY -- see the `ablate-*` features in
                        // Cargo.toml. Skipping the insert corrupts the barcode
                        // (pivots can no longer be resolved) and is only ever
                        // used to price this one call by diffing wall time.
                        //
                        // It is also skipped -- legitimately -- when the whole
                        // population already lives in `apparent_table`, which
                        // was bulk-built before this loop. That is the same
                        // content the sequential insert would have produced:
                        // `PivotLookup` resolves both populations, and
                        // `WALKCROSS` measured the apparent and reduction key
                        // sets to be disjoint.
                        #[cfg(not(feature = "ablate-pivot-insert"))]
                        if _apparent_index.is_none() {
                            pivot_column_index.insert(
                                apparent.get_index(),
                                PivotVal::new(index_column_to_reduce, apparent.get_coefficient()),
                            );
                        }
                        continue;
                    }
                }

                // ---- PHASE1 SPLIT: non-apparent portion of loop_s ----
                //
                // `loop_s` spans the whole chunk loop, so it contains BOTH the
                // apparent fast path (99.89% of iterations on the ASA matrix)
                // AND `pre_pass` (the CPU-only apparent detection, which lives
                // inside the chunk loop). Reporting `loop_s` alone therefore
                // cannot answer "how much would skipping apparent columns
                // save?" -- and adding `pre_pass` on top of it double-counts.
                //
                // This timer runs only on the ~0.1% of columns that actually
                // reduce, and only when CANNS_RIPSER_STATS is set, so the
                // default path is untouched and the added cost is ~2 clocks
                // per reducing column (about 15 ms at ASA scale).
                let na_t0 = if stats {
                    Some(std::time::Instant::now())
                } else {
                    None
                };

                // Scratch buffers, now only for columns that actually reduce.
                working_reduction_column.clear();
                working_coboundary.clear();

                // avoid excessive memory usage
                if working_reduction_column.capacity() > estimated_working_size * 16 {
                    working_reduction_column.shrink_to_fit();
                }
                if working_coboundary.capacity() > estimated_working_size * 16 {
                    working_coboundary.shrink_to_fit();
                }

                working_reduction_column.push(column_to_reduce_entry);

                // Fast path: locate the pivot with a linear scan (no heap). Only if
                // that pivot collides with an existing one do we materialize the
                // coboundary heap and actually reduce. This avoids building and
                // draining a full coboundary heap for the (common) columns that pair
                // off immediately with zero reductions.
                let lookup = PivotLookup::new(pivot_column_index, _apparent_index);
                walk_stats::count_query();
                let t_scan0 = if stats {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let scan_pivot = self.scan_coboundary_pivot(
                    column_to_reduce_entry,
                    dim,
                    &lookup,
                    index_column_to_reduce,
                    &mut cofacet_buf,
                );
                let t_lookup = std::time::Instant::now();
                let needs_reduction = scan_pivot.get_index() != -1 && {
                    let hit = lookup.contains(scan_pivot.get_index(), index_column_to_reduce);
                    if hit {
                        walk_stats::count_hit();
                    }
                    hit
                };
                walk_stats::note_lookup_ns(t_lookup.elapsed().as_nanos() as u64);
                if let Some(t) = t_scan0 {
                    stat_scan_s += t.elapsed().as_secs_f64();
                }
                walk_stats::note_scan_column();

                let mut pivot = if needs_reduction {
                    walk_stats::note_slow_column();
                    let t_fill0 = if stats {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    if one_pass_cofacet {
                        // ONE PASS. The scan above already visited every
                        // cofacet of this column, under exactly the threshold
                        // filter `fill_coboundary_heap` applies, in exactly
                        // the same order -- so the heap can be built from the
                        // scan's own buffer instead of enumerating the whole
                        // coboundary a second time. This is what ripser++
                        // does (`cofacet_entries`).
                        //
                        // The heap content is identical, so the barcode is
                        // identical; the only thing that changes is how many
                        // times the coboundary is walked.
                        for c in cofacet_buf.iter() {
                            working_coboundary.push(*c);
                        }
                    } else {
                        self.fill_coboundary_heap(
                            column_to_reduce_entry,
                            dim,
                            &mut working_coboundary,
                        );
                    }
                    walk_stats::note_heap_fill(working_coboundary.len());
                    let p = self.get_pivot(&mut working_coboundary);
                    if let Some(t) = t_fill0 {
                        stat_fill_s += t.elapsed().as_secs_f64();
                    }
                    p
                } else {
                    scan_pivot
                };

                let t_reduce0 = if stats {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                loop {
                    if pivot.get_index() != -1 {
                        let t_lookup2 = std::time::Instant::now();
                        let got = lookup.get(pivot.get_index(), index_column_to_reduce);
                        walk_stats::note_lookup_ns(t_lookup2.elapsed().as_nanos() as u64);
                        if let Some(stored) = got {
                            stat_rounds += 1;
                            stat_col_rounds_cur += 1;
                            if working_coboundary.len() > stat_heap_max {
                                stat_heap_max = working_coboundary.len();
                            }
                            if working_reduction_column.len() > stat_redheap_max {
                                stat_redheap_max = working_reduction_column.len();
                            }
                            let (index_column_to_add, other_coeff) = stored.into();
                            // Perform matrix reduction
                            if self.verbose {
                                eprintln!("DEBUG: Found pivot collision! pivot_idx={}, current_coeff={}, stored_coeff={}", 
                                     pivot.get_index(), pivot.get_coefficient(), other_coeff);
                            }

                            // Use unsafe to optimize modular inverse calculation
                            let inv = unsafe {
                                *self
                                    .multiplicative_inverse
                                    .get_unchecked(other_coeff as usize)
                            };
                            let prod = (pivot.get_coefficient() as i32) * (inv as i32);
                            let factor_mod = modp(prod as CoefficientT, modulus);
                            let mut factor = modulus - factor_mod;
                            factor = modp(factor, modulus);

                            if self.verbose {
                                eprintln!("DEBUG: Modular inverse calculation: other_coeff={}, inv={}, current_coeff={}, prod={}, factor_mod={}, factor={}, modulus={}",
                                     other_coeff, inv, pivot.get_coefficient(), prod, factor_mod, factor, self.modulus);
                            }

                            debug_assert!(factor != 0, "factor should not be 0");

                            if self.verbose {
                                eprintln!(
                                    "DEBUG: Before add_coboundary_soa: working_coboundary.len()={}",
                                    working_coboundary.len()
                                );
                            }

                            let t_push0 = if stats {
                                Some(std::time::Instant::now())
                            } else {
                                None
                            };
                            self.add_coboundary_soa(
                                &reduction_matrix_soa,
                                columns_to_reduce,
                                index_column_to_add,
                                factor,
                                dim,
                                &mut working_reduction_column,
                                &mut working_coboundary,
                            );
                            if let Some(t) = t_push0 {
                                stat_push_s += t.elapsed().as_secs_f64();
                            }

                            if self.verbose {
                                eprintln!(
                                    "DEBUG: After add_coboundary_soa: working_coboundary.len()={}",
                                    working_coboundary.len()
                                );
                            }

                            let t_pop0 = if stats {
                                Some(std::time::Instant::now())
                            } else {
                                None
                            };
                            pivot = self.get_pivot(&mut working_coboundary);
                            if let Some(t) = t_pop0 {
                                stat_pop_s += t.elapsed().as_secs_f64();
                            }

                            if self.verbose {
                                eprintln!(
                                    "DEBUG: New pivot after reduction: idx={}, coeff={}",
                                    pivot.get_index(),
                                    pivot.get_coefficient()
                                );
                            }
                        } else {
                            // Found a persistence pair
                            let death = pivot.get_diameter();
                            if self.verbose {
                                eprintln!(
                                    "DEBUG: Found pair birth={}, death={}, dim={}",
                                    diameter, death, dim
                                );
                            }

                            if death > diameter * self.ratio {
                                if self.verbose {
                                    eprintln!(
                                        "DEBUG: Recording persistence pair [{}, {}] for dim={}",
                                        diameter, death, dim
                                    );
                                }
                                births_and_deaths.push(diameter);
                                births_and_deaths.push(death);

                                // Compute representative cocycle if requested
                                if do_cocycles {
                                    if self.verbose {
                                        eprintln!("DEBUG: About to compute cocycles for finite pair, dim={}, working_column size={}",
                                              dim, working_reduction_column.len());
                                    }

                                    // Use the same cocycle computation as the original C++ ripser
                                    let mut cocycle = working_reduction_column.clone();
                                    let mut thiscocycle = Vec::new();

                                    // Process working_reduction_column following original C++ logic
                                    loop {
                                        let pivot = self.get_pivot(&mut cocycle);
                                        if pivot.get_index() == -1 {
                                            break;
                                        }

                                        if dim == 1 {
                                            // For H1: add vertices of the edge simplex
                                            let (v_i, v_j) =
                                                self.get_edge_vertices(pivot.get_index(), self.n);
                                            thiscocycle.push(v_i as i32);
                                            thiscocycle.push(v_j as i32);
                                        } else {
                                            // For higher dimensions: add all vertices of the simplex
                                            let vertices = self.get_simplex_vertices(
                                                pivot.get_index(),
                                                dim,
                                                self.n,
                                            );
                                            for vertex in vertices {
                                                thiscocycle.push(vertex as i32);
                                            }
                                        }

                                        // Add normalized coefficient
                                        let normalized_coeff =
                                            self.normalize_coefficient(pivot.get_coefficient());
                                        thiscocycle.push(normalized_coeff as i32);

                                        cocycle.pop();
                                    }

                                    if !thiscocycle.is_empty() {
                                        cocycles.push(thiscocycle);
                                    }
                                }
                            } else if self.verbose {
                                eprintln!(
                                    "DEBUG: Skipping pair [{}, {}] for dim={} due to ratio filter",
                                    diameter, death, dim
                                );
                            }

                            pivot_column_index.insert(
                                pivot.get_index(),
                                PivotVal::new(index_column_to_reduce, pivot.get_coefficient()),
                            );
                            walk_stats::note_reduction_pivot(
                                pivot.get_index(),
                                index_column_to_reduce,
                            );

                            // Store reduction column in SoA format
                            self.pop_pivot(&mut working_reduction_column);
                            loop {
                                let e = self.pop_pivot(&mut working_reduction_column);
                                if e.get_index() == -1 {
                                    break;
                                }
                                debug_assert!(e.get_coefficient() > 0);
                                reduction_matrix_soa.push_back(
                                    e.get_diameter(),
                                    e.get_index(),
                                    e.get_coefficient(),
                                );
                            }
                            break;
                        }
                    } else {
                        // Infinite persistence pair
                        births_and_deaths.push(diameter);
                        births_and_deaths.push(f32::INFINITY);

                        // Ensure cocycles are extracted for infinite intervals too (consistent with C++)
                        if do_cocycles {
                            if self.verbose {
                                eprintln!("DEBUG: About to compute cocycles for infinite pair, dim={}, working_column size={}", 
                                      dim, working_reduction_column.len());
                            }

                            // Use the same cocycle computation as the original C++ ripser
                            let mut cocycle = working_reduction_column.clone();
                            let mut thiscocycle = Vec::new();

                            // Process working_reduction_column following original C++ logic
                            loop {
                                let pivot = self.get_pivot(&mut cocycle);
                                if pivot.get_index() == -1 {
                                    break;
                                }

                                if dim == 1 {
                                    // For H1: add vertices of the edge simplex
                                    let (v_i, v_j) =
                                        self.get_edge_vertices(pivot.get_index(), self.n);
                                    thiscocycle.push(v_i as i32);
                                    thiscocycle.push(v_j as i32);
                                } else {
                                    // For higher dimensions: add all vertices of the simplex
                                    let vertices =
                                        self.get_simplex_vertices(pivot.get_index(), dim, self.n);
                                    for vertex in vertices {
                                        thiscocycle.push(vertex as i32);
                                    }
                                }

                                // Add normalized coefficient
                                let normalized_coeff =
                                    self.normalize_coefficient(pivot.get_coefficient());
                                thiscocycle.push(normalized_coeff as i32);

                                cocycle.pop();
                            }

                            if !thiscocycle.is_empty() {
                                cocycles.push(thiscocycle);
                            }
                        }
                        break;
                    }
                }
                if let Some(t) = t_reduce0 {
                    stat_reduce_s += t.elapsed().as_secs_f64();
                }
                if stat_col_rounds_cur > stat_col_rounds_max {
                    stat_col_rounds_max = stat_col_rounds_cur;
                }
                stat_col_rounds_cur = 0;
                if let Some(t) = na_t0 {
                    stat_nonapparent_loop_s += t.elapsed().as_secs_f64();
                    stat_nonapparent_cols += 1;
                }
            }
        }

        // WALKREPLAY -- runs AFTER the real walk has already produced its
        // barcode, so it cannot affect correctness and cannot warm the caches
        // the real loop saw.
        //
        // The question: the walk costs ~17.7 s (total_s minus gpu_pre minus
        // sharded build), but WALKPROBE says its four memory patterns together
        // cost 1.5 s. Either the cost is intrinsic to those operations on THESE
        // buffers, or it lives in the loop scaffolding around them. Four
        // replays over the same real buffers separate the two:
        //
        //   both    : load columns_to_reduce, push a bounds slot, load gpu_pre
        //   no_appr : the same without the gpu_pre load
        //   no_bnds : the same without the bounds push
        //   neither : the two loads only
        //
        // Absolute numbers here are slightly optimistic (the buffers are warm
        // by now); the DIFFERENCE between the variants is the useful part.
        #[cfg(feature = "walk-probe")]
        if _apparent_index.is_some() || apparent_enabled {
            let n = columns_to_reduce.len();
            let t = std::time::Instant::now();
            let mut sink = 0f32;
            {
                let mut b: Vec<u32> = Vec::with_capacity(n);
                for i in 0..n {
                    let c = &columns_to_reduce[i];
                    b.push(i as u32);
                    let a = &gpu_pre.as_slice()[i];
                    if a.get_index() != -1 {
                        sink += c.get_diameter();
                    }
                }
            }
            let d_both = t.elapsed().as_secs_f64();

            let t = std::time::Instant::now();
            {
                let mut b: Vec<u32> = Vec::with_capacity(n);
                for i in 0..n {
                    b.push(i as u32);
                    sink += columns_to_reduce[i].get_diameter();
                }
            }
            let d_noappr = t.elapsed().as_secs_f64();

            let t = std::time::Instant::now();
            {
                let mut sink2 = 0f32;
                for i in 0..n {
                    sink2 += columns_to_reduce[i].get_diameter();
                    if gpu_pre.as_slice()[i].get_index() != -1 {
                        sink2 += 1.0;
                    }
                }
            }
            let d_nobnds = t.elapsed().as_secs_f64();

            let t = std::time::Instant::now();
            {
                let mut sink2 = 0f32;
                for i in 0..n {
                    sink2 += columns_to_reduce[i].get_diameter();
                    sink2 += gpu_pre.as_slice()[i].get_diameter();
                }
            }
            let d_neither = t.elapsed().as_secs_f64();

            eprintln!(
                "WALKREPLAY n={n} both={d_both:.3} no_apparent={d_noappr:.3} \
                 no_bounds={d_nobnds:.3} loads_only={d_neither:.3} sink={sink:.3}"
            );
        }

        let loop_s = loop_t0.elapsed().as_secs_f64();
        let pc_after = proc_counters();
        if stats {
            // Page faults cost nothing to read and are the one stall source
            // that needs no hardware counters to see. `perf` is unavailable
            // here, but /proc is not.
            eprintln!(
                "LOOPFAULT minflt={} majflt={} utime={} stime={} iters={} \
                 apparent_iters={} chunk={} chunk_len={}",
                pc_after.0.wrapping_sub(pc_before.0),
                pc_after.1.wrapping_sub(pc_before.1),
                pc_after.2.wrapping_sub(pc_before.2),
                pc_after.3.wrapping_sub(pc_before.3),
                iters,
                apparent_iters,
                apparent_chunk,
                columns_to_reduce.len()
            );
        }

        let _apparent_for_stats = if use_gpu_pre {
            Some(gpu_pre.as_slice())
        } else {
            None
        };
        walk_stats::report(dim, _apparent_for_stats);
        // `stat_apparent` is only incremented inside the CHUNKED CPU pre-pass,
        // so it reads 0 whenever the GPU pre-pass supplied the results -- which
        // is every large run. Taking that 0 as fact cost real time here: it
        // looked like dim=1 had no apparent pairs at all, when the counter was
        // simply not wired to that path. Report the GPU path's real count.
        //
        // The sharded build already counts the entries it inserted, so reuse
        // that when it ran. Otherwise count here, which costs one pass over
        // the 4.6 GB `gpu_pre` -- acceptable only because this whole block is
        // behind CANNS_RIPSER_STATS, which is off by default.
        let apparent_pairs_reported = if use_gpu_pre {
            match apparent_table.as_ref() {
                Some(t) => t.present_count(),
                None => gpu_pre
                    .as_slice()
                    .iter()
                    .filter(|c| c.get_index() != -1)
                    .count(),
            }
        } else {
            stat_apparent as usize
        };
        if stats {
            eprintln!(
                "STATS dim={dim} columns={} apparent_pairs={apparent_pairs_reported} \
                 pre_pass_s={:.6} total_s={:.6} share={:.4} cofacets_per_col={:.3} \
                 gpu_pre_s={:.6} densify_s={:.6} sharded_pivot_build_s={:.6} sharded_pivot={} \
                 loop_s={:.3} nonapparent_loop_s={:.3} nonapparent_cols={} \
                 scan_s={:.3} fill_s={:.3} reduce_s={:.3} push_s={:.3} pop_s={:.3} rounds={} heap_max={} redheap_max={} col_rounds_max={} \
                 bd_pairs={} pivots={} reserve_bound={} pre_loop_s={:.6} reserve_s={:.6} \
                 ratio_filter_survives={:.6} gpu_err=\"{}\"",
                columns_to_reduce.len(),
                stat_pre_pass,
                compute_t0.elapsed().as_secs_f64(),
                stat_pre_pass / compute_t0.elapsed().as_secs_f64().max(1e-12),
                apparent_stats::cofacets_per_column(),
                stat_gpu_s,
                stat_densify,
                stat_sharded_build,
                _apparent_index.is_some(),
                // How much barcode this dimension actually produced, and what
                // fraction of the enumerated columns survived the ratio
                // filter. Both are needed to size anything on
                // `births_and_deaths`; see report S22.11.
                loop_s,
                // PHASE1 SPLIT. `loop_s` = apparent fast path + pre_pass +
                // this. Skipping apparent columns could save at most
                // loop_s - nonapparent_loop_s - pre_pass_s, and nothing at all
                // if that difference is small.
                stat_nonapparent_loop_s,
                stat_nonapparent_cols,
                // The three-way split of `nonapparent_loop_s`, plus the number
                // of reduction rounds. `scan_s + fill_s + reduce_s` must
                // account for `nonapparent_loop_s` up to the clock-read
                // overhead; anything larger is unattributed work.
                stat_scan_s,
                stat_fill_s,
                stat_reduce_s,
                stat_push_s,
                stat_pop_s,
                stat_rounds,
                stat_heap_max,
                stat_redheap_max,
                stat_col_rounds_max,
                births_and_deaths.len() / 2,
                // RESERVE GATE: when the sharded table exists the map is
                // reserved for `columns - apparent_count`, so the final length
                // must never exceed that bound. Printing both makes the
                // invariant checkable on ANY input, not only the one where
                // WALKCROSS happened to measure the two key sets disjoint.
                pivot_column_index.len(),
                _reserve,
                stat_pre_loop_s,
                stat_reserve_s,
                apparent_pairs_reported as f64 / columns_to_reduce.len().max(1) as f64,
                gpu_err
            );
        }

        // Final progress update
        if should_report_progress {
            if let Some(ref callback) = progress_callback {
                pyo3::Python::attach(|py| {
                    let _ = callback.call1(
                        py,
                        (
                            columns_to_reduce.len(),
                            columns_to_reduce.len(),
                            format!("Completed H{} computation", dim),
                        ),
                    );
                });
            }
        }

        if self.verbose {
            eprintln!(
                "DEBUG: compute_pairs dim={} complete, processed {} columns, found {} pairs",
                dim,
                columns_to_reduce.len(),
                births_and_deaths.len() / 2
            );
        }
    }
}
