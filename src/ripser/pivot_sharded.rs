//! Parallel-built sharded open-addressing table for the APPARENT pivot
//! population -- the 286,263,529-entry bulk that dominates the ASA run.
//!
//! # Why this exists
//!
//! The reduction walk discovers apparent pairs in a pre-pass (P1/P2, GPU or
//! CPU) and then inserts each one into a single `FxHashMap` **sequentially**,
//! interleaved with the walk. On the ASA matrix those inserts are ~22 s of a
//! 45.7 s run: they touch 2.86e8 random cache lines across a ~9 GB table.
//!
//! # Why it is safe to build ahead of the walk
//!
//! Two measured facts (`WALKCROSS` in `reduction.rs`):
//!   * apparent keys are unique -- 286,263,529 present, 0 duplicates;
//!   * apparent keys never collide with reduction-created pivot keys --
//!     `collide_with_apparent = 0` at dim 1 and dim 2.
//!
//! So the apparent population is a pure function of (column, dim) with no
//! dependence on walk order, and a bulk build produces exactly the map the
//! sequential inserts would have produced. `PivotLookup` still applies the
//! sequential-visibility rule `pos < before` on read, so an entry that has not
//! "happened yet" stays invisible.
//!
//! The second fact is handled exactly, by construction: `PivotLookup` compares
//! the two positions and lets the later one win, which is precisely the
//! overwrite the single map performed.
//!
//! The first fact is a genuine precondition, not something the code can repair.
//! With duplicate apparent keys the original map answers a query at column c
//! with "the last insert strictly before c", so for a query landing between two
//! duplicate inserts it returns the EARLIER value. One value per key cannot
//! express that; `build_from_columns` keeps the later column, which agrees
//! everywhere except between the duplicates. Measured at 0 duplicates in
//! 286,263,529 keys, so the disagreement is unreachable on this workload --
//! `duplicate_keys_keep_the_later_column` pins the behaviour if it ever is not.
//!
//! # When this table must NOT be used
//!
//! Assembly of the next dimension reads `pivot_column_index` directly,
//! bypassing `PivotLookup`. `compute_pairs` already carries the flag that
//! says whether that will happen (`pivot_map_needed_for_assembly`, i.e.
//! `dim < dim_max`) and the build is gated on it. Do not remove that gate.
//!
//! # Why sharded
//!
//! Six alternatives were built and measured before this one; all are recorded
//! in `~/pivot_micro/src/bin/sharded_hash.rs` and in the report:
//!
//!   sorted array + high-bit directory : lookups 18.7 us each on the real
//!                                        clustered keys (59.1 s -> 84.6 s)
//!   GPU atomicCAS build, 2^29 slots   : 100% GPU for 12 min, never finished
//!   GPU atomicCAS build, 2^30 slots   : 100% GPU for 7 min, never finished
//!   narrow the entry 24 -> 16 B      : no measurable effect
//!   Bloom filter in front            : 3.2% slower, and 99.75% of assembly's
//!                                        probes answer "is a pivot", so an
//!                                        accelerator that only helps negatives
//!                                        cannot help here
//!   prune to the queryable set       : circular, the walk needs the map to
//!                                        decide what to scan
//!
//! # Measured (real ASA keys, 16 cores, `sharded_hash`)
//!
//! ```text
//!   A  sequential FxHashMap          22.10 s   77.2 ns/entry   57.1 ns/query
//!   B1 count -> scatter -> per-shard  5.63 s   19.7 ns/entry   71.2 ns/query
//!   B2 count -> atomic claim          4.02 s   13.9 ns/entry   71.8 ns/query   <-- this
//!   B3 B2 with key+value interleaved 4.55 s                     (see below)
//! ```
//!
//! B3 is the instructive failure. Packing key and value into one 16 B slot so
//! the value store hits the line the CAS just fetched sounds like it removes a
//! DRAM miss per insert -- and it is *slower by 13%*. The value write then
//! depends on the CAS result, so the two accesses become two serialised round
//! trips instead of two concurrent misses. B2 keeps the arrays separate: both
//! addresses are known before the CAS is issued, so the misses overlap.
//!
//! That also explains the remaining 13.9 ns/entry. It is latency, not
//! bandwidth: 2.86e8 inserts move ~55 GB, which at this machine's memory
//! bandwidth is well under a second. 16 cores x ~1 outstanding insert each is
//! simply not enough memory-level parallelism to hide ~200 ns of DRAM latency.
//!
//! # Memory
//!
//! Slots are sized exactly as hashbrown would size an `FxHashMap::reserve`
//! (`next_pow2(n * 8 / 7)`), so this table and the map it replaces both come to
//! 2^29 slots at the ASA scale: 8.6 GB either way. Nothing is traded away.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::ripser::types::{DiameterEntryT, IndexT, PivotVal};

/// Sentinel for an unclaimed slot. `EntryT` reserves -1 for "no entry", so no
/// real apparent key can collide with it.
const EMPTY: u64 = u64::MAX;

const SHARD_BITS: u32 = 8;
/// Keep shards large enough that the per-shard load stays well under 1.
const MIN_SLOTS_PER_SHARD: usize = 1 << 17;
/// Above this the table would exceed 2^32 slots and the 48-bit position field
/// would start to matter; refuse rather than try to allocate 64 GB.
const MAX_ENTRIES: usize = 1 << 31;

/// splitmix64 finaliser. The high bits are what select the shard and the low
/// bits pick the offset inside it, so the two must be independent.
#[inline(always)]
fn mix(k: IndexT) -> u64 {
    let mut z = (k as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline(always)]
fn next_pow2(mut n: usize) -> usize {
    n = n.saturating_sub(1);
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n.wrapping_add(1)
}

pub struct ShardedPivotTable {
    /// `EMPTY` when unclaimed. Atomic because `build_from_columns` claims slots
    /// in parallel; relaxed loads on the read side compile to a plain `mov`.
    keys: Vec<AtomicU64>,
    /// `PivotVal` packed, same slot index as `keys`.
    vals: Vec<AtomicU64>,
    shard_mask: u64,
    off_mask: usize,
    present: usize,
}

impl ShardedPivotTable {
    /// Bulk-build from the pre-pass output. `columns[pos]` is the apparent
    /// cofacet found for column `pos`; entries whose index is -1 are not
    /// apparent and are skipped.
    ///
    /// Returns `None` when the request is out of range, so the caller can fall
    /// back to the sequential path rather than abort.
    pub fn build_from_columns(columns: &[DiameterEntryT]) -> Option<Self> {
        let present = columns.iter().filter(|c| c.get_index() != -1).count();
        if present > MAX_ENTRIES {
            return None;
        }
        // hashbrown's own sizing rule, so this table and the FxHashMap it
        // replaces occupy the same number of slots.
        let slots = next_pow2(present.saturating_mul(8).saturating_div(7)).max(1 << 10);
        let shards = (slots / MIN_SLOTS_PER_SHARD).clamp(1, 1 << SHARD_BITS);
        let sps = slots / shards;

        let table = Self {
            keys: (0..slots).map(|_| AtomicU64::new(EMPTY)).collect(),
            vals: (0..slots).map(|_| AtomicU64::new(0)).collect(),
            shard_mask: (shards - 1) as u64,
            off_mask: sps - 1,
            present,
        };

        #[cfg(feature = "parallel")]
        table.fill_parallel(columns);
        #[cfg(not(feature = "parallel"))]
        table.fill_sequential(columns);
        Some(table)
    }

    #[inline(always)]
    fn claim(
        keys: &[AtomicU64],
        vals: &[AtomicU64],
        shard_mask: u64,
        off_mask: usize,
        key: IndexT,
        val: PivotVal,
    ) {
        let h = mix(key);
        let sbase = ((h >> (64 - SHARD_BITS)) & shard_mask) as usize * (off_mask + 1);
        let mut off = (h as usize) & off_mask;
        let v = val.to_packed();
        loop {
            let s = sbase + off;
            match keys[s].compare_exchange_weak(
                EMPTY,
                key as u64,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Exactly one thread claims a slot, and both addresses are
                    // known before the CAS is issued, so the two misses overlap.
                    vals[s].store(v, Ordering::Relaxed);
                    return;
                }
                Err(seen) => {
                    if seen == key as u64 {
                        // Duplicate key. Measured at 0 in 286,263,529 on the
                        // real data, so this is unreachable in practice; see
                        // the module docs for why "keep the later column" is
                        // the closest a one-value-per-key table can get.
                        // fetch_max is order-independent, which a plain store
                        // would not be under a parallel build.
                        vals[s].fetch_max(v, Ordering::Relaxed);
                        return;
                    }
                    off = (off + 1) & off_mask;
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    fn fill_parallel(&self, columns: &[DiameterEntryT]) {
        use rayon::prelude::*;
        // Disjoint-by-construction writes into `vals` are done with Relaxed
        // atomics rather than a raw-pointer wrapper, so this needs no unsafe.
        columns.par_iter().enumerate().for_each(|(pos, c)| {
            let key = c.get_index();
            if key == -1 {
                return;
            }
            Self::claim(
                &self.keys,
                &self.vals,
                self.shard_mask,
                self.off_mask,
                key,
                PivotVal::new(pos, c.get_coefficient()),
            );
        });
    }

    #[cfg(not(feature = "parallel"))]
    fn fill_sequential(&self, columns: &[DiameterEntryT]) {
        for (pos, c) in columns.iter().enumerate() {
            let key = c.get_index();
            if key == -1 {
                continue;
            }
            Self::claim(
                &self.keys,
                &self.vals,
                self.shard_mask,
                self.off_mask,
                key,
                PivotVal::new(pos, c.get_coefficient()),
            );
        }
    }

    /// Entries fed to the build. Free: the count is taken anyway to size the
    /// table, and `STATS apparent_pairs` has no other source on the GPU
    /// pre-pass path (see `compute_pairs`).
    pub fn present_count(&self) -> usize {
        self.present
    }

    /// O(1) lookup, no allocation, no lock.
    #[inline]
    pub fn get(&self, key: IndexT) -> Option<PivotVal> {
        let h = mix(key);
        let sbase = ((h >> (64 - SHARD_BITS)) & self.shard_mask) as usize * (self.off_mask + 1);
        let mut off = (h as usize) & self.off_mask;
        loop {
            let s = sbase + off;
            let k = self.keys[s].load(Ordering::Relaxed) as IndexT;
            // EMPTY first, not key first: EMPTY is `u64::MAX`, which IS -1 as
            // an IndexT, so testing `k == key` first would let a query for -1
            // match the first empty slot it lands on.
            if k == EMPTY as IndexT {
                return None;
            }
            if k == key {
                return Some(PivotVal::from_packed(self.vals[s].load(Ordering::Relaxed)));
            }
            off = (off + 1) & self.off_mask;
        }
    }

    /// Number of claimed slots. Only used by tests -- a full scan of 8.6 GB.
    #[cfg(test)]
    pub fn occupied(&self) -> usize {
        self.keys
            .iter()
            .filter(|k| k.load(Ordering::Relaxed) != EMPTY)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ripser::types::CoefficientT;

    fn cols(n: usize) -> Vec<DiameterEntryT> {
        (0..n)
            .map(|i| {
                if i % 7 == 0 {
                    DiameterEntryT::new(1.0, -1, 0)
                } else {
                    // Strictly increasing: no duplicate keys. See
                    // `duplicate_keys_keep_the_later_column` for what a
                    // duplicate does and why the real data has none.
                    DiameterEntryT::new(
                        i as f32,
                        (i as IndexT) * 7919 - 500_000,
                        (i % 251) as CoefficientT,
                    )
                }
            })
            .collect()
    }

    #[test]
    fn roundtrips_every_present_column() {
        let c = cols(50_000);
        let t = ShardedPivotTable::build_from_columns(&c).expect("built");
        assert_eq!(
            t.occupied(),
            c.iter().filter(|x| x.get_index() != -1).count()
        );
        for (pos, e) in c.iter().enumerate() {
            if e.get_index() == -1 {
                assert!(t.get(-1).is_none());
                continue;
            }
            let got = t.get(e.get_index()).expect("present");
            assert_eq!(got.pos(), pos, "wrong position for {}", e.get_index());
            assert_eq!(got.coeff(), e.get_coefficient());
        }
    }

    /// The blind spot that let the empty-table regression through: both
    /// production sources (`gpu_pre` on the GPU path, `apparent_pre` on the
    /// hoisted CPU path) must yield the SAME table for the same column set,
    /// and a source that was never filled must be obvious rather than silently
    /// producing an empty table.
    #[test]
    fn both_production_sources_give_the_same_table() {
        let c = cols(20_000);
        let apparent_count = c.iter().filter(|x| x.get_index() != -1).count();

        // Source 1: shaped like `gpu_pre` -- every slot present, with
        // non-apparent slots carrying the (-1, 0) sentinel.
        let gpu_like = c.clone();
        // Source 2: shaped like the hoisted CPU `apparent_pre` -- same array,
        // produced by detect_apparent instead of a kernel.
        let apparent_pre_like = c.clone();

        let a = ShardedPivotTable::build_from_columns(&gpu_like).expect("gpu-like built");
        let b = ShardedPivotTable::build_from_columns(&apparent_pre_like).expect("apparent built");

        assert_eq!(
            a.present_count(),
            apparent_count,
            "gpu_pre source lost apparent entries"
        );
        assert_eq!(
            b.present_count(),
            apparent_count,
            "apparent_pre source lost apparent entries"
        );
        assert_eq!(a.present_count(), b.present_count());
        for e in c.iter().filter(|x| x.get_index() != -1) {
            assert!(a.get(e.get_index()).is_some());
            assert!(b.get(e.get_index()).is_some());
        }
    }

    /// The exact failure mode: a source that was never filled produces an empty
    /// table that still answers `get(..) == None` for everything, so nothing
    /// downstream complains -- the barcode is simply wrong. Assert the shape of
    /// that trap so it is visible in review.
    #[test]
    fn an_unfilled_source_yields_an_empty_table_not_a_partial_one() {
        let unfilled: Vec<DiameterEntryT> = Vec::new(); // exactly what `apparent_pre` looks like on the GPU path
        let t = ShardedPivotTable::build_from_columns(&unfilled).expect("built");
        assert_eq!(t.present_count(), 0);
        assert_eq!(t.get(42), None);
        // A correctly filled source of the same shape would NOT be empty.
        let filled = cols(1_000);
        let ok = ShardedPivotTable::build_from_columns(&filled).expect("built");
        assert!(ok.present_count() > 0);
    }

    #[test]
    fn absent_keys_are_absent() {
        let c = cols(5_000);
        let t = ShardedPivotTable::build_from_columns(&c).expect("built");
        // -1 is load-bearing, not decorative: EMPTY is u64::MAX, which IS -1 as
        // an IndexT. A `get` that tests the key before the empty marker lets a
        // -1 query match the first empty slot it lands on, and `get(-1)` is
        // reachable -- the walk asks `contains(cofacet.get_index(), ..)` and a
        // missing cofacet reports index -1.
        for k in [-1i64, -2, 0, 7_777_777, 500_001, i64::MAX, i64::MIN] {
            assert!(t.get(k).is_none(), "phantom hit for {k}");
        }
    }

    #[test]
    fn duplicate_keys_keep_the_later_column() {
        // Two columns report the same cofacet; "insert in column order" means
        // the second one overwrites the first.
        //
        // This is NOT fully equivalent to the original single map: that map
        // answers a query at column c with "the last insert before c", so for a
        // query between the two columns it would return the FIRST one. A
        // one-value-per-key table cannot express that. WALKCROSS measured 0
        // duplicate apparent keys in 286,263,529 on the ASA matrix, so on real
        // input the distinction never arises; this test pins what does happen
        // if it ever does.
        let c = vec![
            DiameterEntryT::new(1.0, 42, 3),
            DiameterEntryT::new(2.0, -1, 0),
            DiameterEntryT::new(3.0, 42, 7),
        ];
        let t = ShardedPivotTable::build_from_columns(&c).expect("built");
        assert_eq!(t.occupied(), 1);
        let got = t.get(42).expect("present");
        assert_eq!(got.pos(), 2);
        assert_eq!(got.coeff(), 7);
    }

    #[test]
    fn tiny_input_does_not_panic() {
        for n in 0..64 {
            let c = cols(n);
            let t = ShardedPivotTable::build_from_columns(&c).expect("built");
            assert_eq!(
                t.occupied(),
                c.iter().filter(|x| x.get_index() != -1).count()
            );
        }
    }

    /// The whole justification for pre-loading is that the split lookup in
    /// `PivotLookup` reproduces the single `FxHashMap` it replaced. Simulate
    /// that map directly -- insert every apparent entry, then every reduction
    /// entry, in ascending column order, and answer `pos < before` queries from
    /// it -- and require the split form to agree at every query column.
    ///
    /// The `OVERLAP` seed deliberately forces apparent and reduction populations
    /// onto the SAME keys. `WALKCROSS` measured that never happens on the ASA
    /// matrix (collide_with_apparent = 0 at both dims), so this is the case the
    /// real data cannot test but a different input could.
    #[test]
    fn split_lookup_matches_single_map_including_collisions() {
        use rustc_hash::FxHashMap;

        for overlap in [0usize, 3, 11] {
            // column -> (apparent key or -1, reduction key or -1).
            //
            // Apparent keys are STRICTLY increasing, i.e. no duplicates. That is
            // not a convenience: with duplicates the original map holds "the
            // last insert before `before`", which needs per-query history that
            // a one-value-per-key table cannot represent. WALKCROSS measured 0
            // duplicates in 286,263,529 apparent keys, so on real input the two
            // agree everywhere; `duplicate_keys_keep_the_later_column` pins the
            // documented behaviour when they do occur.
            let mut plan: Vec<(IndexT, IndexT)> = Vec::new();
            let mut next_apparent: IndexT = -500;
            for pos in 0..4_000usize {
                let a = if pos % 5 == 0 {
                    -1
                } else {
                    next_apparent += 1 + ((pos * 2654435761) % 3) as IndexT;
                    next_apparent
                };
                let r = if pos % 3 == 0 {
                    -1
                } else {
                    // Reduction keys overlap the apparent range on purpose:
                    // `WALKCROSS` measured them disjoint on the ASA matrix, so
                    // this is the case the real data cannot test but another
                    // input could.
                    ((pos * 40503) % 2013) as IndexT - 1006 + overlap as IndexT
                };
                plan.push((a, r));
            }

            let apparent_cols: Vec<DiameterEntryT> = plan
                .iter()
                .map(|&(a, _)| {
                    if a == -1 {
                        DiameterEntryT::new(0.0, -1, 0)
                    } else {
                        DiameterEntryT::new(0.0, a, 1)
                    }
                })
                .collect();
            let table = ShardedPivotTable::build_from_columns(&apparent_cols).expect("built");

            // Reference: one map holding exactly what the sequential walk had
            // inserted by the time it reached the lookup sites of column c.
            let mut reference: FxHashMap<IndexT, PivotVal> = FxHashMap::default();

            let mut st = Stats::default();
            for (before, &(a, r)) in plan.iter().enumerate() {
                // The walk's lookup sites run BEFORE this column records
                // anything, so the reference map must not contain column
                // `before`'s entries yet.
                //
                // Probe the full key space, not just this column's own keys:
                // the walk queries coboundary keys it never planned for.
                for k in -1200i64..1200 {
                    probe_at(&table, &reference, k, before, &mut st);
                }
                probe_at(&table, &reference, a, before, &mut st);
                probe_at(&table, &reference, r, before, &mut st);

                if a != -1 {
                    reference.insert(a, PivotVal::new(before, 1));
                }
                if r != -1 {
                    reference.insert(r, PivotVal::new(before, 2));
                }
            }

            let Stats {
                queries,
                hits,
                disagreements,
                first_bad,
            } = st;
            // Guard against a VACUOUS pass: an empty probe set, or one that
            // never hits, would make the equality below true for free. This
            // test is the only thing standing between "the split lookup matches
            // the single map" and "the split lookup was never exercised".
            assert!(
                queries > 3_000_000,
                "only {queries} queries: probe set too small"
            );
            assert!(hits > 100_000, "only {hits} hits: nothing exercised");
            assert_eq!(
                disagreements, 0,
                "overlap={overlap}: split != single map; first: {first_bad:?}"
            );
        }
    }

    #[derive(Default)]
    struct Stats {
        queries: usize,
        hits: usize,
        disagreements: usize,
        first_bad: Option<String>,
    }

    fn probe_at(
        table: &ShardedPivotTable,
        reference: &rustc_hash::FxHashMap<IndexT, PivotVal>,
        key: IndexT,
        before: usize,
        st: &mut Stats,
    ) {
        if key == -1 {
            return;
        }
        st.queries += 1;
        // The REAL resolver, not a copy of it: a mirrored copy would let the
        // test keep passing after the implementation drifted.
        let got = crate::ripser::core::reduction::PivotLookup::new(reference, Some(table))
            .get(key, before);
        let want = reference.get(&key).copied();
        if got.is_some() {
            st.hits += 1;
        }
        if got.map(|v| (v.pos(), v.coeff())) != want.map(|v| (v.pos(), v.coeff())) {
            st.disagreements += 1;
            if st.first_bad.is_none() {
                st.first_bad = Some(format!(
                    "key={key} before={before} split={got:?} map={want:?}"
                ));
            }
        }
    }
}
