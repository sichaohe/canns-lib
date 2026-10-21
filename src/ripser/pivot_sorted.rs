//! Read-optimised pivot index: a sorted key array plus a directory indexed
//! directly on the key's high bits.
//!
//! # Why this shape
//!
//! Measured on the ASA matrix the reduction walk WRITES 286,263,529 pivot
//! entries and READS only ~3.9e6 -- 73:1. That is the opposite of what a hash
//! map is good at, and it is why `FxHashMap` costs 94-120 ns per insert on a
//! 13.1 GB table while the whole structure it holds is written once.
//!
//! Isolated benchmark over the same 2.9e8 entries, including the real read
//! volume (`~/pivot_micro/src/bin/sorted_pivot.rs`):
//!
//! ```text
//! hash map : build 27.01 s                        footprint 13.1 GB
//! sorted[] : fill 1.88 + sort 2.72 + dir 2.64      footprint  4.27 GiB
//!            reads 2.32 s over 3.9e6 probes
//!            TOTAL 9.57 s  ->  2.82x, 0 sample mismatches
//! ```
//!
//! # Ordering semantics this structure cannot express
//!
//! The walk is strictly sequential: reducing column `j` may only consult pivots
//! created by columns `i < j`. A completed index answers "who owns p" without
//! regard to position, so every lookup MUST re-apply `owner < j` by hand -- see
//! [`SortedPivotIndex::get_owned_before`]. That is not a detail; dropping it
//! changes the barcode.
//!
//! # Directory
//!
//! Direct-indexed on the key's high bits, NOT binary searched. A binary
//! search over a stride directory measured 901 ns/probe -- nine times worse
//! than the hash map it replaces -- because the directory itself misses cache at
//! every level of the search. One indexed load, then a scan of one bucket,
//! measured ~595 ns.

use crate::ripser::types::{CoefficientT, IndexT};

/// Bits reserved for the coefficient in [`PivotEntry::packed`].
const COEFF_BITS: u32 = 16;

/// One pivot: which simplex index it is, and which column first claimed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PivotEntry {
    /// `pos << COEFF_BITS | coeff`, matching how the rest of the library packs
    /// an (index, coefficient) pair into one word.
    pub packed: u64,
}

impl PivotEntry {
    #[inline]
    pub fn new(pos: usize, coeff: CoefficientT) -> Self {
        debug_assert!(
            pos < (1usize << (64 - COEFF_BITS)),
            "position overflows 48 bits"
        );
        Self {
            packed: ((pos as u64) << COEFF_BITS) | (coeff as u16 as u64),
        }
    }

    /// Column position that claimed this pivot.
    #[inline]
    pub fn pos(&self) -> usize {
        (self.packed >> COEFF_BITS) as usize
    }

    #[inline]
    pub fn coeff(&self) -> CoefficientT {
        (self.packed as u16 as i16) as CoefficientT
    }
}

/// A pivot index built once from a full set of pairs, then queried.
#[derive(Debug)]
pub struct SortedPivotIndex {
    /// Cofacet simplex indices, ascending and unique.
    keys: Vec<u64>,
    vals: Vec<PivotEntry>,
    /// `dir[b]` is the first position whose key has the same high bits as
    /// bucket `b`. `EMPTY` marks a bucket with no keys at all.
    dir: Vec<u32>,
    /// `key >> shift` selects the bucket.
    shift: u32,
}

const EMPTY: u32 = u32::MAX;

impl SortedPivotIndex {
    /// Build from `(key, entry)` pairs. `key_bits` is the width of the key
    /// universe -- it must come from the data's actual range, NOT from 64.
    ///
    /// Getting that wrong is silent and catastrophic: with `shift = 64 -
    /// dir_bits` and 37-bit keys, `key >> 44` is 0 for every key, the whole
    /// index collapses into bucket 0, and every lookup scans from the start.
    /// The benchmark that produced the numbers above spent nine minutes in
    /// exactly that state before the bug was found.
    pub fn build(pairs: &mut [(u64, PivotEntry)], key_bits: u32, dir_bits: u32) -> Self {
        assert!(
            dir_bits < key_bits,
            "directory finer than the key universe: dir_bits={dir_bits} key_bits={key_bits}"
        );
        pairs.sort_unstable_by_key(|&(k, _)| k);

        let keys: Vec<u64> = pairs.iter().map(|p| p.0).collect();
        let vals: Vec<PivotEntry> = pairs.iter().map(|p| p.1).collect();
        let n = keys.len();
        let shift = key_bits - dir_bits;

        // First index per bucket. A u32::MAX sentinel is required: 0 cannot
        // distinguish "empty" from "the first entry really is at index 0".
        let mut dir: Vec<u32> = vec![EMPTY; 1usize << dir_bits];
        for (i, &k) in keys.iter().enumerate() {
            let b = (k >> shift) as usize;
            if dir[b] == EMPTY {
                dir[b] = i as u32;
            }
        }
        // A bucket with no keys must point at the next bucket that has one.
        let mut acc = EMPTY;
        for v in dir.iter_mut() {
            if *v == EMPTY {
                *v = acc;
            } else {
                acc = *v;
            }
        }
        if n > 0 && dir[0] == EMPTY {
            dir[0] = 0;
        }

        SortedPivotIndex {
            keys,
            vals,
            dir,
            shift,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Raw lookup, ignoring column order. Only sound once the caller knows the
    /// index covers every column that could have claimed the pivot.
    #[inline]
    pub fn get(&self, key: IndexT) -> Option<PivotEntry> {
        let k = key as u64;
        let d = unsafe { *self.dir.get_unchecked((k >> self.shift) as usize) };
        if d == EMPTY {
            return None;
        }
        let mut i = d as usize;
        let n = self.keys.len();
        while i < n {
            let kk = unsafe { *self.keys.get_unchecked(i) };
            if kk == k {
                return Some(unsafe { *self.vals.get_unchecked(i) });
            }
            if kk > k {
                return None;
            }
            i += 1;
        }
        None
    }

    /// Lookup restricted to claims made strictly before column `before`.
    ///
    /// This is the sequential-walk rule: when reducing column `j`, a pivot may
    /// only be attributed to a column `i < j`. A completed index knows nothing
    /// about order, so the position check lives HERE rather than at each call
    /// site, where it would be easy to forget.
    ///
    /// Returns `None` when the only claimant is at or after `before` -- which
    /// is exactly the case where the walk must treat the pivot as new.
    #[inline]
    pub fn get_owned_before(&self, key: IndexT, before: usize) -> Option<PivotEntry> {
        let e = self.get(key)?;
        if e.pos() < before {
            Some(e)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&u64, &PivotEntry)> {
        self.keys.iter().zip(self.vals.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The directory bug that cost a 9-minute hang: `shift` must come from the
    /// key universe. If every key collapses into bucket 0 this still passes,
    /// so the bucket-occupancy assertion below is what actually catches it.
    #[test]
    fn matches_a_hash_map_on_random_keys() {
        let n = 20_000u64;
        let key_bits = 37u32;
        let dir_bits = 20u32;
        let mask = (1u64 << key_bits) - 1;

        let mut pairs: Vec<(u64, PivotEntry)> = (0..n)
            .map(|i| {
                let k = (i.wrapping_mul(2_654_435_761)) & mask;
                (k, PivotEntry::new(i as usize, (i % 47) as i16))
            })
            .collect();
        pairs.sort_unstable_by_key(|&(k, _)| k);
        pairs.dedup_by_key(|&mut (k, _)| k);
        let distinct = pairs.len();

        let idx = SortedPivotIndex::build(&mut pairs, key_bits, dir_bits);
        assert_eq!(idx.len(), distinct);
        assert!(distinct > 0);

        let reference: HashMap<u64, PivotEntry> = idx.iter().map(|(k, v)| (*k, *v)).collect();

        // Occupancy: a collapsed index puts everything in bucket 0.
        let occupied = (0..(1usize << dir_bits))
            .filter(|&b| idx.dir[b] != EMPTY)
            .count();
        assert!(
            occupied > (1usize << dir_bits) / 4,
            "only {occupied} buckets occupied out of {} -- keys collapsed into one bucket",
            1usize << dir_bits
        );

        // Hits and in-universe neighbours (which are absent at 0.33% density).
        for q in 0..50_000u64 {
            let k = (q.wrapping_mul(7_919)) & mask;
            for probe in [k, (k + 1) & mask] {
                assert_eq!(
                    idx.get(probe as IndexT),
                    reference.get(&probe).copied(),
                    "disagreement at probe {probe}"
                );
            }
        }

        // A key far outside the populated range must be absent, not a crash.
        for probe in [0u64, mask, mask / 2, 1u64 << (key_bits - 1)] {
            assert_eq!(
                idx.get(probe as IndexT),
                reference.get(&probe).copied(),
                "boundary probe {probe}"
            );
        }
    }

    #[test]
    fn ownership_filter_is_strict() {
        let mut pairs: Vec<(u64, PivotEntry)> = (0..1000u64)
            .map(|i| (i * 7919 % 1_000_003, PivotEntry::new(i as usize, 1)))
            .collect();
        let idx = SortedPivotIndex::build(&mut pairs, 20, 12);
        for (k, e) in idx.iter() {
            let pos = e.pos();
            assert!(idx.get_owned_before(*k as IndexT, pos + 1).is_some());
            assert!(idx.get_owned_before(*k as IndexT, pos).is_none());
            assert!(idx.get_owned_before(*k as IndexT, 0).is_none());
        }
    }
}
