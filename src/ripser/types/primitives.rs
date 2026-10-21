// Core types matching C++ implementation
pub type ValueT = f32;
pub type IndexT = i64;
pub type CoefficientT = i16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    LowerTriangular,
    UpperTriangular,
}

// Number of low bits reserved for the coefficient when packing (index,
// coefficient) into a single word. Must match binomial.rs's overflow bound.
const NUM_COEFFICIENT_BITS: u32 = 8;
const COEFFICIENT_MASK: i64 = (1 << NUM_COEFFICIENT_BITS) - 1;

// Entry type for homology computation.
//
// Index and coefficient are packed into a single 64-bit word (index in the high
// bits, coefficient in the low 8 bits) to match ripser's compact representation.
// This halves the footprint of DiameterEntryT (24 -> 16 bytes) and, since the
// reduction working columns are binary heaps of these entries, materially cuts
// the memory traffic of the reduction hot loop. Coefficients are always
// normalised into [0, modulus) with modulus < 2^8, so they fit the low bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryT {
    data: i64,
}

impl EntryT {
    #[inline(always)]
    pub fn new(index: IndexT, coefficient: CoefficientT) -> Self {
        Self {
            data: (index << NUM_COEFFICIENT_BITS) | ((coefficient as i64) & COEFFICIENT_MASK),
        }
    }

    #[inline(always)]
    pub fn get_index(&self) -> IndexT {
        // Arithmetic shift preserves the sign of the -1 sentinel index.
        self.data >> NUM_COEFFICIENT_BITS
    }

    #[inline(always)]
    pub fn get_coefficient(&self) -> CoefficientT {
        (self.data & COEFFICIENT_MASK) as CoefficientT
    }

    #[inline(always)]
    pub fn set_coefficient(&mut self, coefficient: CoefficientT) {
        self.data = (self.data & !COEFFICIENT_MASK) | ((coefficient as i64) & COEFFICIENT_MASK);
    }
}

// Diameter-entry pair
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiameterEntryT {
    pub diameter: ValueT,
    pub entry: EntryT,
}

impl DiameterEntryT {
    pub fn new(diameter: ValueT, index: IndexT, coefficient: CoefficientT) -> Self {
        Self {
            diameter,
            entry: EntryT::new(index, coefficient),
        }
    }

    pub fn get_diameter(&self) -> ValueT {
        self.diameter
    }

    pub fn get_index(&self) -> IndexT {
        self.entry.get_index()
    }

    pub fn get_coefficient(&self) -> CoefficientT {
        self.entry.get_coefficient()
    }

    pub fn set_coefficient(&mut self, coefficient: CoefficientT) {
        self.entry.set_coefficient(coefficient);
    }
}

impl Eq for DiameterEntryT {}

// Ordering for priority queue (greater diameter or smaller index)
impl Ord for DiameterEntryT {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // For BinaryHeap (max-heap) to behave like C++ min-heap:
        // - smaller diameter should be considered "greater"
        // - on tie, larger index should be considered "greater"
        // Diameters here are finite, non-negative distances, so their IEEE-754
        // bit patterns compare as unsigned integers in the same order as the
        // floats. This is a single integer compare vs. total_cmp's bit-twiddling
        // and is called on every heap sift, so it matters for the hot loop.
        other
            .diameter
            .to_bits()
            .cmp(&self.diameter.to_bits())
            .then_with(|| self.get_index().cmp(&other.get_index()))
    }
}

impl PartialOrd for DiameterEntryT {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }

    /// Direct `<=`, the test `BinaryHeap::sift_up` actually performs.
    ///
    /// The default `PartialOrd::le` goes through `partial_cmp`, which here
    /// wraps the three-way `cmp` in an `Option` and then matches on it. That
    /// comparison runs on EVERY sift of EVERY push -- 45,156,214 pushes at
    /// dim 1 and 78,969,010 at dim 2 on the ASA matrix -- so the `Option`
    /// round trip is worth removing. The two branches below are the same
    /// order `cmp` implements, written out:
    ///
    ///   * different diameters: `cmp` compares `other` against `self`, so
    ///     `self <= other` exactly when `other.diameter < self.diameter`
    ///     (smaller diameter is the "greater" one, which is what makes the
    ///     max-heap behave like the C++ min-heap).
    ///   * equal diameters: `cmp` compares `self.index` against
    ///     `other.index`, so `self <= other` exactly when
    ///     `self.index <= other.index`.
    #[inline(always)]
    fn le(&self, other: &Self) -> bool {
        let a = self.diameter.to_bits();
        let b = other.diameter.to_bits();
        if a != b {
            b < a
        } else {
            self.get_index() <= other.get_index()
        }
    }
}

// Working column type for matrix reduction.
//
// The default is the hand-rolled `CoboundaryHeap`, whose `pop` does the
// CLASSIC top-down sift-down and stops as soon as the element fits -- the
// same algorithm `std::priority_queue::pop` (what ripser++ uses) implements.
//
// `std::collections::BinaryHeap`, which this replaces, pops with
// `sift_down_to_bottom`: it moves the hole all the way to the BOTTOM of the
// tree and then sifts the element back up. Fewer comparisons, but it ALWAYS
// touches the bottom of the tree -- and the reduction's coboundary heap
// reaches 9-11 million entries (148-185 MB), so the bottom of the tree is a
// cold cache line on every one of the 23 levels.
//
// MEASURED on the ASA matrix (2026-10-20, three interleaved reps, the
// barcode byte-for-byte identical in every run):
//
//     inf input    11.791 s -> 11.024 s   (-6.5%)
//     capped input 11.806 s -> 10.988 s   (-6.9%)
//
// UNVERIFIED IN THE CURRENT DELIVERY: that measurement has no table behind
// it. See the same note in Cargo.toml, and the re-measurement in
// `benchmarks/ripser/harness/unified_ablation.py`.
//
// The `binary-heap` feature restores `std::collections::BinaryHeap` (it is the
// opt-OUT now). The two heaps are verified to yield the SAME pop order by
// `coboundary_heap::pop_order_matches_binary_heap`.
#[cfg(not(feature = "binary-heap"))]
pub type WorkingT = crate::ripser::types::coboundary_heap::CoboundaryHeap;
#[cfg(feature = "binary-heap")]
pub type WorkingT = std::collections::BinaryHeap<DiameterEntryT>;

// Diameter-index pair
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiameterIndexT {
    pub diameter: ValueT,
    pub index: IndexT,
}

impl DiameterIndexT {
    pub fn new(diameter: ValueT, index: IndexT) -> Self {
        Self { diameter, index }
    }

    pub fn get_diameter(&self) -> ValueT {
        self.diameter
    }

    pub fn get_index(&self) -> IndexT {
        self.index
    }
}

impl Ord for DiameterIndexT {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Use total_cmp for consistent ordering without NaN panic paths
        other
            .diameter
            .total_cmp(&self.diameter)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for DiameterIndexT {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for DiameterIndexT {}

// Index-diameter pair for sparse matrices
#[derive(Debug, Clone, Copy)]
pub struct IndexDiameterT {
    pub index: IndexT,
    pub diameter: ValueT,
}

impl IndexDiameterT {
    pub fn new(index: IndexT, diameter: ValueT) -> Self {
        Self { index, diameter }
    }

    pub fn get_index(&self) -> IndexT {
        self.index
    }

    pub fn get_diameter(&self) -> ValueT {
        self.diameter
    }
}

/// Compact value stored in the pivot map: which column first made a simplex a
/// pivot, and with what coefficient.
///
/// The map holds one entry per enumerated column. Measured on the real ASA
/// matrix at dim=2 that is 286,562,199 entries, whose keys are 3-simplex
/// indices spanning [0, C(1200,4)) -- 86 billion distinct values. hashbrown
/// therefore allocates 2^29 buckets, and the entry width decides the whole
/// bill:
///
/// ```text
/// (i64, (usize, i16)) = 24 B/entry -> 13.1 GB
/// (i64, PivotVal)     = 16 B/entry ->  9.1 GB
/// ```
///
/// An isolated micro-benchmark of exactly 286,562,199 inserts prices the
/// difference: 64.9 ns/insert against a ~8.0 ns floor for the same inserts
/// into a cache-resident table, i.e. 57 of every 65 ns is memory-system cost
/// rather than hashing. Narrower entries mean fewer bytes touched.
///
/// The position is packed into the high 48 bits rather than narrowed to u32 so
/// no runtime range check is needed: 2^48 columns is 2.8e14, far beyond any
/// enumeration this library can produce (a u32 position would overflow at
/// C(3000,3) = 4.5e9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PivotVal {
    /// `pos << 16 | (coeff as u16)`.
    packed: u64,
}

impl PivotVal {
    /// Bits reserved for the coefficient in the low half.
    const COEFF_BITS: u32 = 16;
    /// Largest position that survives the shift.
    const MAX_POS: usize = (1usize << (64 - Self::COEFF_BITS)) - 1;

    #[inline]
    pub fn new(pos: usize, coeff: CoefficientT) -> Self {
        debug_assert!(
            pos <= Self::MAX_POS,
            "pivot column position {pos} exceeds the 48-bit field"
        );
        Self {
            packed: ((pos as u64) << Self::COEFF_BITS) | (coeff as u16 as u64),
        }
    }

    #[inline]
    pub fn pos(&self) -> usize {
        (self.packed >> Self::COEFF_BITS) as usize
    }

    #[inline]
    pub fn coeff(&self) -> CoefficientT {
        // Sign-extend: the low 16 bits hold a two's-complement coefficient.
        (self.packed as u16 as i16) as CoefficientT
    }

    /// Raw `pos << 16 | coeff`. Used by the sharded pivot table, which stores
    /// the value in its own atomic word and must not re-pack it per probe.
    #[inline]
    pub fn to_packed(&self) -> u64 {
        self.packed
    }

    #[inline]
    pub fn from_packed(packed: u64) -> Self {
        Self { packed }
    }
}

impl From<(usize, CoefficientT)> for PivotVal {
    #[inline]
    fn from((pos, coeff): (usize, CoefficientT)) -> Self {
        PivotVal::new(pos, coeff)
    }
}

impl From<PivotVal> for (usize, CoefficientT) {
    #[inline]
    fn from(v: PivotVal) -> Self {
        (v.pos(), v.coeff())
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    /// The GPU hands the apparent population back as three separate arrays.
    /// If `DiameterEntryT` were 16 bytes AND had a fixed field order, the
    /// kernel could write one record per column and the host could read the
    /// pinned buffer in place -- no host-side copy at all. This test pins
    /// down what the layout actually is, so the decision is made on a
    /// measurement rather than on Rust's default-repr rules.
    #[test]
    fn apparent_entry_layout() {
        // The GPU writes the apparent population as 16-byte records that the
        // host reads IN PLACE as `&[DiameterEntryT]` (see `ApparentRec` in
        // kernel/ripser_kernels.cu). That only works while the two layouts
        // agree bit for bit, and Rust's default repr does NOT promise an
        // order -- so the order is pinned HERE, by assertion.
        //
        // If a compiler ever reorders the fields, this test fails and the
        // .cu has to be updated. The alternative -- trusting the default
        // repr -- would silently read the wrong bytes and produce a wrong
        // barcode, which is the failure mode this whole project keeps
        // running into.
        assert_eq!(std::mem::size_of::<DiameterEntryT>(), 16);
        assert_eq!(std::mem::align_of::<DiameterEntryT>(), 8);
        assert_eq!(std::mem::size_of::<EntryT>(), 8);

        let e = DiameterEntryT::new(1.5f32, 7i64, 3i16);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &e as *const DiameterEntryT as *const u8,
                std::mem::size_of::<DiameterEntryT>(),
            )
        };
        // bytes 0..8: the packed entry -- coefficient in the low 8 bits,
        // index above it; bytes 8..12: the diameter.
        //
        // The last four bytes are PADDING and are deliberately NOT asserted:
        // Rust does not zero a struct's padding, so a value built on the
        // stack carries whatever was there. On the GPU path the padding
        // comes from the device's zeroed allocation, so it is always zero --
        // but nothing reads it, and the kernel does not write it either.
        assert_eq!(
            &bytes[0..12],
            &[
                3, 7, 0, 0, 0, 0, 0, 0, // (7 << 8) | 3, little endian
                0, 0, 192, 63, // 1.5f32
            ]
        );

        // And the accessors must round-trip the packed form, because that is
        // what the kernel writes.
        assert_eq!(e.get_index(), 7);
        assert_eq!(e.get_coefficient(), 3);
        assert_eq!(e.get_diameter(), 1.5f32);
    }
}

#[cfg(test)]
mod ordering_tests {
    use super::{DiameterEntryT, EntryT};

    /// `PartialOrd::le` is hand-written (see the impl above) so that the
    /// heap sift does not pay for an `Option<Ordering>` round trip. It must
    /// agree with `cmp` EXACTLY, or the reduction would pick different
    /// pivots and the barcode would change. This test checks the agreement
    /// on a deterministic spread of values, including the ties that decide
    /// the pivot.
    #[test]
    fn le_agrees_with_cmp() {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..4000u64 {
            // Deliberately coarse diameters so ties are common: a tie is
            // resolved by the index, which is the branch that is easy to
            // get backwards.
            let d = ((next() % 7) as f32) * 0.25;
            let idx = (next() % 1000) as i64;
            a.push(DiameterEntryT::new(d, idx, 1));
            b.push(DiameterEntryT::new(
                d,
                (next() % 1000) as i64,
                (next() % 47) as i16,
            ));
        }
        for i in 0..a.len() {
            for j in 0..a.len() {
                let x = a[i];
                let y = b[j];
                // `le` is "not Greater" and `ge` is "not Less". They are NOT
                // complements: when `cmp` is Equal -- same diameter AND
                // same index, differing only in the coefficient, which the
                // packed entry allows -- BOTH are true.
                let want_le = x.cmp(&y) != std::cmp::Ordering::Greater;
                let want_ge = x.cmp(&y) != std::cmp::Ordering::Less;
                assert_eq!(x.le(&y), want_le, "le disagrees with cmp for {x:?} / {y:?}");
                assert_eq!(x.ge(&y), want_ge, "ge disagrees with cmp for {x:?} / {y:?}");
            }
        }
    }

    /// The index is packed into the low bits of the entry, so the heap
    /// comparison has to shift it back out. Check that the packed form and
    /// the extracted index agree, and that the -1 sentinel survives.
    #[test]
    fn packed_index_round_trips() {
        // The packed form is `(index << 8) | coefficient`, so it is only
        // lossless for index < 2^55 -- which is exactly the contract
        // `MAX_SIMPLEX_INDEX` states. Testing beyond it would be testing
        // an unsupported input, not the packing.
        for raw in [-1i64, 0, 1, 255, 256, 1_000_003, (1i64 << 55) - 1] {
            let e = EntryT::new(raw, 47);
            assert_eq!(e.get_index(), raw);
        }
    }
}
