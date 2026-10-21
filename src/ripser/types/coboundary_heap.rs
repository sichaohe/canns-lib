use crate::ripser::types::{CoefficientT, DiameterEntryT, IndexT, ValueT};

/// A binary max-heap whose `pop` uses the CLASSIC top-down sift-down with an
/// early exit.
///
/// WHY NOT `std::collections::BinaryHeap`. Its `pop` calls
/// `sift_down_to_bottom`: it moves the hole all the way to the BOTTOM of the
/// tree and then sifts the element back up. That is the bottom-up variant --
/// fewer comparisons, but it ALWAYS touches the bottom of the tree, then
/// walks back up. `std::priority_queue::pop` (what ripser++ uses) does the
/// classic top-down sift and stops as soon as the element is in place, so it
/// usually never reaches the cold part of the tree.
///
/// The coboundary heap in the reduction reaches 9.3M (dim 2) and 11.5M (dim 1)
/// entries -- 148 MB and 185 MB -- so the bottom of the tree is a guaranteed
/// cache miss per level, and there are 23 levels. Measured on the ASA matrix
/// (2026-10-20), the reduction's `get_pivot` phase -- which is dominated by
/// pops -- was 1.69x slower than ripser++'s at dim 1 and 1.42x at dim 2, while
/// the push phase was on par. That is the signature of a pop that walks the
/// whole tree.
///
/// The comparison is the same ordering `DiameterEntryT`'s `Ord` implements:
/// smaller diameter is "greater" (so the max-heap yields the smallest
/// diameter first), and on a tie the larger index is "greater".
#[derive(Debug, Clone, Default)]
pub struct CoboundaryHeap {
    data: Vec<DiameterEntryT>,
}

impl CoboundaryHeap {
    #[inline(always)]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            data: Vec::with_capacity(n),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.data.clear();
    }

    #[inline(always)]
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// `true` when `a` is GREATER than `b` in `DiameterEntryT`'s `Ord`, i.e.
    /// when `a` belongs nearer the TOP of the heap.
    ///
    /// `DiameterEntryT`'s `Ord` compares `other.diameter` against
    /// `self.diameter` -- the SMALLER diameter is the "greater" one, which is
    /// what makes the max-heap yield the smallest diameter first, and on a tie
    /// the LARGER index is the "greater" one.
    #[inline(always)]
    fn above(a: &DiameterEntryT, b: &DiameterEntryT) -> bool {
        let da = a.diameter.to_bits();
        let db = b.diameter.to_bits();
        if da != db {
            // smaller diameter is "greater". The bit patterns of
            // non-negative floats compare in the same order as the floats.
            da < db
        } else {
            // on a tie the larger index is "greater".
            a.get_index() > b.get_index()
        }
    }

    #[inline(always)]
    pub fn push(&mut self, item: DiameterEntryT) {
        self.data.push(item);
        self.sift_up(self.data.len() - 1);
    }

    #[inline]
    fn sift_up(&mut self, mut pos: usize) {
        let item = self.data[pos];
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if Self::above(&item, &self.data[parent]) {
                self.data[pos] = self.data[parent];
                pos = parent;
            } else {
                break;
            }
        }
        self.data[pos] = item;
    }

    /// The classic sift-down: move the hole down, always towards the larger
    /// child, and stop as soon as the element fits.
    #[inline]
    pub fn pop(&mut self) -> Option<DiameterEntryT> {
        let last = self.data.pop()?;
        if self.data.is_empty() {
            return Some(last);
        }
        let item = self.data[0];
        let mut pos = 0usize;
        let n = self.data.len();
        loop {
            let l = 2 * pos + 1;
            if l >= n {
                break;
            }
            // pick the child that is nearer the top
            let mut child = l;
            let r = l + 1;
            if r < n && Self::above(&self.data[r], &self.data[l]) {
                child = r;
            }
            if Self::above(&self.data[child], &last) {
                self.data[pos] = self.data[child];
                pos = child;
            } else {
                break;
            }
        }
        self.data[pos] = last;
        Some(item)
    }

    /// The element the pop would return, without removing it.
    #[inline(always)]
    pub fn peek(&self) -> Option<&DiameterEntryT> {
        self.data.first()
    }
}

// Keep the unused-import linter quiet about the types that only appear in the
// doc comments above.
#[allow(dead_code)]
type _Unused = (CoefficientT, IndexT, ValueT);

#[cfg(test)]
mod tests {
    use super::CoboundaryHeap;
    use crate::ripser::types::DiameterEntryT;
    use std::collections::BinaryHeap;

    /// The hand-rolled heap must yield the SAME pop order as
    /// `std::collections::BinaryHeap` for the same input, or the reduction
    /// would pick different pivots and the barcode would change.
    ///
    /// The comparison is a total order on (diameter, index) with the
    /// coefficient packed into the entry, so the two heaps agree on the
    /// ORDER of the elements even though they disagree on the order of
    /// EQUAL elements -- which is fine, because equal elements are
    /// interchangeable for the reduction.
    #[test]
    fn pop_order_matches_binary_heap() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut items = Vec::new();
        for _ in 0..20_000 {
            // Coarse diameters so ties on the diameter are common, and a
            // spread of indices so the tie-break is exercised.
            let d = ((next() % 64) as f32) * 0.125;
            let idx = (next() % 5000) as i64;
            items.push(DiameterEntryT::new(d, idx, ((next() % 47) as i16) + 1));
        }

        let mut mine = CoboundaryHeap::with_capacity(items.len());
        let mut theirs = BinaryHeap::with_capacity(items.len());
        for it in &items {
            mine.push(*it);
            theirs.push(*it);
        }
        assert_eq!(mine.len(), theirs.len());

        // Compare the SEQUENCE OF (diameter, index) pairs, not the raw
        // entries: equal elements may be returned in either order.
        let mut a = Vec::new();
        while let Some(e) = mine.pop() {
            a.push((e.get_diameter().to_bits(), e.get_index()));
        }
        let mut b = Vec::new();
        while let Some(e) = theirs.pop() {
            b.push((e.get_diameter().to_bits(), e.get_index()));
        }
        assert_eq!(a.len(), b.len());
        assert!(a == b, "the two heaps disagree on the pop order");
    }

    /// A heap must pop its elements in non-increasing "greatness" order, i.e.
    /// the smallest diameter first (and, on a tie, the largest index first).
    #[test]
    fn pops_in_reduction_order() {
        let mut h = CoboundaryHeap::with_capacity(8);
        h.push(DiameterEntryT::new(2.0, 10, 1));
        h.push(DiameterEntryT::new(1.0, 20, 1));
        h.push(DiameterEntryT::new(1.0, 5, 1));
        h.push(DiameterEntryT::new(3.0, 1, 1));
        let order: Vec<(f32, i64)> = std::iter::from_fn(|| h.pop())
            .map(|e| (e.get_diameter(), e.get_index()))
            .collect();
        assert_eq!(order, vec![(1.0, 20), (1.0, 5), (2.0, 10), (3.0, 1)]);
    }
}
