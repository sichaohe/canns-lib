use crate::ripser::matrix::traits::{DistanceMatrix, EdgeProvider, HasCofacets, VertexBirth};
use crate::ripser::types::{
    CoefficientT, DiameterEntryT, DiameterIndexT, IndexDiameterT, IndexT, ValueT,
};
use crate::ripser::utils::BinomialCoeffTable;
#[cfg(feature = "parallel")]
use rayon::prelude::ParallelSliceMut;

// Note: Removed ripser compatibility traits since we're eliminating ripser_old

// Sparse distance matrix
#[derive(Debug, Clone)]
pub struct SparseDistanceMatrix {
    pub neighbors: Vec<Vec<IndexDiameterT>>,
    pub vertex_births: Vec<ValueT>,
    pub num_edges: IndexT,
}

impl SparseDistanceMatrix {
    pub fn from_dense<M: DistanceMatrix>(mat: &M, threshold: ValueT) -> Self {
        let n = mat.size();
        let mut neighbors = vec![Vec::<IndexDiameterT>::new(); n];
        let mut num_edges = 0;

        // Use parallel processing for large matrices
        if n > 1000 {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;

                // Process all edges in parallel
                let edges: Vec<(usize, usize, ValueT)> = (0..n)
                    .into_par_iter()
                    .flat_map(|i| {
                        (i + 1..n)
                            .filter_map(|j| {
                                let v = mat.get(i, j);
                                if v <= threshold && v.is_finite() {
                                    Some((i, j, v))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();

                // Count edges for each vertex
                let mut edge_counts = vec![0; n];
                for &(i, j, _) in &edges {
                    edge_counts[i] += 1;
                    edge_counts[j] += 1;
                }

                // Pre-allocate vectors
                for i in 0..n {
                    neighbors[i].reserve(edge_counts[i]);
                }

                // Add edges
                for (i, j, v) in edges {
                    neighbors[i].push(IndexDiameterT::new(j as IndexT, v));
                    neighbors[j].push(IndexDiameterT::new(i as IndexT, v));
                    num_edges += 1;
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..n {
                    for j in (i + 1)..n {
                        let v = mat.get(i, j);
                        if v <= threshold && v.is_finite() {
                            neighbors[i].push(IndexDiameterT::new(j as IndexT, v));
                            neighbors[j].push(IndexDiameterT::new(i as IndexT, v));
                            num_edges += 1;
                        }
                    }
                }
            }
        } else {
            // Original sequential implementation for small matrices
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = mat.get(i, j);
                    if v <= threshold && v.is_finite() {
                        neighbors[i].push(IndexDiameterT::new(j as IndexT, v));
                        neighbors[j].push(IndexDiameterT::new(i as IndexT, v));
                        num_edges += 1;
                    }
                }
            }
        }

        // Sort neighbors by index for efficient lookup
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            neighbors.par_iter_mut().for_each(|ngh| {
                ngh.sort_by_key(|p| p.index);
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for ngh in &mut neighbors {
                ngh.sort_by_key(|p| p.index);
            }
        }

        Self {
            neighbors,
            vertex_births: vec![0.0; n], // Dense matrices have zero vertex birth times
            num_edges,
        }
    }

    pub fn new(neighbors: Vec<Vec<IndexDiameterT>>, num_edges: IndexT) -> Self {
        let n = neighbors.len();
        Self {
            neighbors,
            vertex_births: vec![0.0; n],
            num_edges,
        }
    }

    pub fn from_coo(
        i: &[i32],
        j: &[i32],
        v: &[f32],
        n_edges: i32,
        n: i32,
        threshold: ValueT,
    ) -> Result<Self, String> {
        let n_usize = n as usize;
        let mut neighbors = vec![vec![]; n_usize];
        let mut vertex_births = vec![0.0; n_usize];
        let mut num_edges = 0;

        for idx in 0..n_edges as usize {
            let i_val = i[idx] as usize;
            let j_val = j[idx] as usize;
            let val = v[idx];

            if val.is_nan() {
                return Err(format!("NaN distance found at sparse matrix index {}", idx));
            }

            if i_val < j_val && val <= threshold {
                neighbors[i_val].push(IndexDiameterT::new(j_val as IndexT, val));
                neighbors[j_val].push(IndexDiameterT::new(i_val as IndexT, val));
                num_edges += 1;
            } else if i_val == j_val {
                vertex_births[i_val] = val;
            }
        }

        // Sort neighbors and remove duplicates
        for neighbor_list in &mut neighbors {
            neighbor_list.sort_unstable_by(|a, b| a.index.cmp(&b.index));
            neighbor_list.dedup_by_key(|p| p.index);
        }

        Ok(Self {
            neighbors,
            vertex_births,
            num_edges,
        })
    }

    pub fn size(&self) -> usize {
        self.neighbors.len()
    }
}

// Compressed sparse matrix for reduction operations - optimized to SoA layout for improved cache performance
#[derive(Debug, Clone)]
pub struct CompressedSparseMatrix<T> {
    bounds: Vec<usize>,
    entries: Vec<T>,
}

impl<T> CompressedSparseMatrix<T> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            bounds: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.bounds.len()
    }

    pub fn append_column(&mut self) {
        self.bounds.push(self.entries.len());
    }

    pub fn push_back(&mut self, e: T) {
        debug_assert!(!self.bounds.is_empty(), "No column to push to");
        self.entries.push(e);
        if let Some(last) = self.bounds.last_mut() {
            *last = self.entries.len();
        }
    }

    pub fn pop_back(&mut self) {
        debug_assert!(!self.bounds.is_empty(), "No column to pop from");
        if !self.entries.is_empty() {
            self.entries.pop();
            if let Some(last) = self.bounds.last_mut() {
                *last = self.entries.len();
            }
        }
    }

    pub fn subrange(&self, index: usize) -> &[T] {
        if index == 0 {
            &self.entries[0..self.bounds[index]]
        } else {
            &self.entries[self.bounds[index - 1]..self.bounds[index]]
        }
    }
}

// Optimized SoA layout version, specifically for DiameterEntryT
//
// `bounds` used to be `Vec<u32>` with ONE ENTRY PER ENUMERATED COLUMN: it
// stored the end offset of that column's entries, so the offset of column
// `i` was `bounds[i]`. `append_column` runs once per enumerated column --
// 286,562,199 times at dim=2 on the ASA matrix -- so the array itself was
// 1.15 GB at dim=2 (403 MB at the T threshold), and every one of those
// writes was a cache line that carried no information: the overwhelming
// majority of the columns are APPARENT pairs, which push no entries at
// all, so their offsets are all equal to the previous column's.
//
// The array is now a list of (column, end offset) pairs, one per NON-EMPTY
// column: 298,670 of them at dim=2, i.e. 2.4 MB instead of 1.15 GB. The
// columns are processed in increasing order, so the pairs are sorted by
// column number and a column's range is found with a binary search. The
// semantics are unchanged: `subrange(c)` returns exactly the entries that
// were pushed while column `c` was being processed, and an empty column
// returns an empty range -- which is what the old dense array gave, since
// `bounds[i] == bounds[i-1]` for a column with no entries.
//
// The binary search is O(log n) in the number of non-empty columns, and
// `subrange` is called once per reduction round (116,926 at dim=2 on the
// ASA matrix), so the extra work is a rounding error.
#[derive(Debug, Clone)]
pub struct OptimizedSparseMatrix {
    bounds: Vec<ColumnBound>,
    diameters: Vec<f32>,
    indices: Vec<IndexT>,
    coefficients: Vec<CoefficientT>,
    /// The column currently being filled, set by `begin_column`.
    current_column: u32,
    /// The column of the last entry in `bounds`, or `NO_COLUMN` when there
    /// is none. Compared against `current_column` to learn whether the
    /// current column already has a bound.
    last_column: u32,
}

/// The end offset of one non-empty column.
#[derive(Debug, Clone, Copy)]
struct ColumnBound {
    column: u32,
    end: u32,
}

/// Sentinel for "no column" -- no real column number can be this large
/// (the enumerated column count is bounded by C(n, 3), and the `u32` field
/// already limits it to 4.29e9 columns).
const NO_COLUMN: u32 = u32::MAX;

impl OptimizedSparseMatrix {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            bounds: Vec::new(),
            diameters: Vec::new(),
            indices: Vec::new(),
            coefficients: Vec::new(),
            current_column: NO_COLUMN,
            last_column: NO_COLUMN,
        }
    }

    /// The number of NON-EMPTY columns (the columns that actually stored a
    /// reduction column). It is not the number of enumerated columns: the
    /// old dense array had one slot per enumerated column, of which the
    /// overwhelming majority were empty.
    pub fn size(&self) -> usize {
        self.bounds.len()
    }

    /// Start filling column `column`. Every entry pushed until the next
    /// `begin_column` belongs to it.
    pub fn begin_column(&mut self, column: usize) {
        debug_assert!(
            column < NO_COLUMN as usize,
            "enumerated column number does not fit the u32 field"
        );
        self.current_column = column as u32;
    }

    #[inline(always)]
    pub fn push_back(&mut self, diameter: f32, index: IndexT, coefficient: CoefficientT) {
        debug_assert!(
            self.current_column != NO_COLUMN,
            "push_back without begin_column"
        );
        debug_assert!(
            self.diameters.len() <= u32::MAX as usize,
            "reduction column entry count exceeds the u32 offset field"
        );
        // The first entry of a column opens its bound; the rest only extend
        // it. An empty column never reaches this point, so it never gets a
        // bound -- which is the whole point of the layout.
        if self.last_column != self.current_column {
            self.bounds.push(ColumnBound {
                column: self.current_column,
                end: self.diameters.len() as u32,
            });
            self.last_column = self.current_column;
        }
        self.diameters.push(diameter);
        self.indices.push(index);
        self.coefficients.push(coefficient);
        if let Some(last) = self.bounds.last_mut() {
            last.end = self.diameters.len() as u32;
        }
    }

    pub fn pop_back(&mut self) {
        if self.diameters.is_empty() {
            return;
        }
        self.diameters.pop();
        self.indices.pop();
        self.coefficients.pop();
        if self.bounds.is_empty() {
            return;
        }
        // Where the last bound's column began.
        let start = if self.bounds.len() >= 2 {
            self.bounds[self.bounds.len() - 2].end as usize
        } else {
            0
        };
        let new_end = self.diameters.len();
        let last = self.bounds.len() - 1;
        if new_end == start {
            // The column just became EMPTY, so its bound goes away with it.
            // The bound before it now ends where the entries end -- it used
            // to end at this column's first entry.
            self.bounds.pop();
            self.last_column = NO_COLUMN;
            if let Some(prev) = self.bounds.last_mut() {
                prev.end = new_end as u32;
            }
        } else {
            self.bounds[last].end = new_end as u32;
        }
    }

    /// The entries pushed while column `column` was being filled.
    ///
    /// A column with no entries returns an empty range, exactly as the old
    /// dense `bounds` array did (`bounds[i] == bounds[i-1]`).
    #[inline(always)]
    pub fn subrange(&self, column: usize) -> (&[f32], &[IndexT], &[CoefficientT]) {
        let c = column as u32;
        let idx = self.bounds.partition_point(|b| b.column < c);
        if idx >= self.bounds.len() || self.bounds[idx].column != c {
            return (
                &self.diameters[0..0],
                &self.indices[0..0],
                &self.coefficients[0..0],
            );
        }
        let start = if idx == 0 {
            0
        } else {
            self.bounds[idx - 1].end as usize
        };
        let end = self.bounds[idx].end as usize;
        (
            &self.diameters[start..end],
            &self.indices[start..end],
            &self.coefficients[start..end],
        )
    }

    pub fn reserve(&mut self, additional: usize) {
        self.diameters.reserve(additional);
        self.indices.reserve(additional);
        self.coefficients.reserve(additional);
    }
}

// Optimized distance calculation functions
#[inline(always)]
pub fn simd_distance_squared(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi) * (xi - yi))
        .sum()
}

#[inline(always)]
pub fn simd_euclidean_distance(x: &[f32], y: &[f32]) -> f32 {
    simd_distance_squared(x, y).sqrt()
}

impl DistanceMatrix for SparseDistanceMatrix {
    fn size(&self) -> usize {
        SparseDistanceMatrix::size(self)
    }

    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }
        // For efficiency, search in the shorter adjacency list.
        let (a, b) = if self.neighbors[i].len() <= self.neighbors[j].len() {
            (i, j)
        } else {
            (j, i)
        };
        let neighbors = &self.neighbors[a];
        match neighbors.binary_search_by_key(&(b as IndexT), |x| x.index) {
            Ok(pos) => neighbors[pos].diameter,
            Err(_) => f32::INFINITY,
        }
    }
}

impl VertexBirth for SparseDistanceMatrix {
    fn vertex_birth(&self, i: IndexT) -> ValueT {
        self.vertex_births[i as usize]
    }
}

impl EdgeProvider for SparseDistanceMatrix {
    fn edges_under_threshold(
        &self,
        threshold: ValueT,
        n: IndexT,
        binom: &BinomialCoeffTable,
    ) -> Vec<DiameterIndexT> {
        let mut edges = Vec::new();

        for i in 0..(n as usize) {
            for neighbor in &self.neighbors[i] {
                let j = neighbor.index as usize;
                if i < j && neighbor.diameter <= threshold {
                    let index = binom.get(j as IndexT, 2) + (i as IndexT);
                    edges.push(DiameterIndexT::new(neighbor.diameter, index));
                }
            }
        }

        // Sort edges by diameter (ascending), then by index (descending) for H0 compatibility
        #[cfg(feature = "parallel")]
        edges.par_sort_unstable_by(|a, b| {
            a.get_diameter()
                .total_cmp(&b.get_diameter())
                .then_with(|| b.get_index().cmp(&a.get_index()))
        });
        #[cfg(not(feature = "parallel"))]
        edges.sort_unstable_by(|a, b| {
            a.get_diameter()
                .total_cmp(&b.get_diameter())
                .then_with(|| b.get_index().cmp(&a.get_index()))
        });

        edges
    }
}

impl HasCofacets for SparseDistanceMatrix {
    // Reuse the dense enumerator: cofacet enumeration only needs distance access.
    type Enumerator<'a> = crate::ripser::matrix::dense::SimplexCoboundaryEnumerator<'a, Self>;

    fn cofacet_enumerator<'a>(
        &'a self,
        simplex: DiameterEntryT,
        dim: IndexT,
        n: IndexT,
        binomial_coeff: &'a BinomialCoeffTable,
        modulus: CoefficientT,
    ) -> Self::Enumerator<'a> {
        crate::ripser::matrix::dense::SimplexCoboundaryEnumerator::new(
            simplex,
            dim,
            n,
            self,
            binomial_coeff,
            modulus,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout this one replaced: one `u32` per ENUMERATED column,
    /// holding the end offset of that column's entries. Kept here as the
    /// reference the sparse layout is checked against -- the two must agree
    /// on every column, including the empty ones.
    struct DenseBounds {
        bounds: Vec<u32>,
        diameters: Vec<f32>,
        indices: Vec<IndexT>,
        coefficients: Vec<CoefficientT>,
    }

    impl DenseBounds {
        fn new(n_columns: usize) -> Self {
            Self {
                bounds: Vec::with_capacity(n_columns),
                diameters: Vec::new(),
                indices: Vec::new(),
                coefficients: Vec::new(),
            }
        }

        fn append_column(&mut self) {
            self.bounds.push(self.diameters.len() as u32);
        }

        fn push_back(&mut self, diameter: f32, index: IndexT, coefficient: CoefficientT) {
            self.diameters.push(diameter);
            self.indices.push(index);
            self.coefficients.push(coefficient);
            if let Some(last) = self.bounds.last_mut() {
                *last = self.diameters.len() as u32;
            }
        }

        fn subrange(&self, column: usize) -> (&[f32], &[IndexT], &[CoefficientT]) {
            let start = if column == 0 {
                0
            } else {
                self.bounds[column - 1] as usize
            };
            let end = self.bounds[column] as usize;
            (
                &self.diameters[start..end],
                &self.indices[start..end],
                &self.coefficients[start..end],
            )
        }
    }

    /// A deterministic sequence of (column, number of entries) pairs that
    /// mixes empty columns, single-entry columns and fat ones.
    fn sample_columns(n_columns: usize) -> Vec<(usize, usize)> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..n_columns)
            .map(|c| {
                let r = next();
                // ~70% of the columns are empty, like the apparent pairs.
                let n = if r % 10 < 7 {
                    0
                } else {
                    1 + (r / 10 % 5) as usize
                };
                (c, n)
            })
            .collect()
    }

    #[test]
    fn subrange_matches_the_old_dense_bounds() {
        let columns = sample_columns(500);
        let mut sparse = OptimizedSparseMatrix::new();
        let mut dense = DenseBounds::new(columns.len());
        let mut next_diameter = 0.5f32;

        for &(column, n_entries) in &columns {
            sparse.begin_column(column);
            dense.append_column();
            for k in 0..n_entries {
                let diameter = next_diameter;
                next_diameter += 1.0;
                let index = (column * 31 + k) as IndexT;
                let coefficient = ((k % 5) + 1) as CoefficientT;
                sparse.push_back(diameter, index, coefficient);
                dense.push_back(diameter, index, coefficient);
            }
        }

        for &(column, _) in &columns {
            let (sd, si, sc) = sparse.subrange(column);
            let (dd, di, dc) = dense.subrange(column);
            assert_eq!(sd, dd, "diameters differ in column {}", column);
            assert_eq!(si, di, "indices differ in column {}", column);
            assert_eq!(sc, dc, "coefficients differ in column {}", column);
        }
    }

    #[test]
    fn empty_columns_have_empty_ranges() {
        let mut m = OptimizedSparseMatrix::new();
        m.begin_column(0);
        m.push_back(1.0, 7, 1);
        m.begin_column(1); // empty
        m.begin_column(2);
        m.push_back(2.0, 9, 1);
        m.push_back(3.0, 10, 2);

        assert_eq!(m.subrange(0).0, &[1.0]);
        assert_eq!(m.subrange(1).0.len(), 0);
        assert_eq!(m.subrange(2).0, &[2.0, 3.0]);
        assert_eq!(m.subrange(2).1, &[9, 10]);
        assert_eq!(m.subrange(2).2, &[1, 2]);
        // A column that was never begun is empty as well.
        assert_eq!(m.subrange(3).0.len(), 0);
        // Two non-empty columns, not three.
        assert_eq!(m.size(), 2);
    }

    #[test]
    fn a_later_non_empty_column_is_found_after_empty_ones() {
        let mut m = OptimizedSparseMatrix::new();
        for column in 0..10 {
            m.begin_column(column);
        }
        m.begin_column(10);
        m.push_back(5.0, 42, 1);
        assert_eq!(m.subrange(10).0, &[5.0]);
        assert_eq!(m.subrange(9).0.len(), 0);
        assert_eq!(m.subrange(0).0.len(), 0);
    }

    #[test]
    fn pop_back_removes_the_bound_of_a_column_that_became_empty() {
        let mut m = OptimizedSparseMatrix::new();
        m.begin_column(0);
        m.push_back(1.0, 1, 1);
        m.begin_column(1);
        m.push_back(2.0, 2, 1);
        assert_eq!(m.size(), 2);
        // Popping the only entry of column 1 empties it, so its bound goes
        // away -- and column 0's bound now ends where the entries end.
        m.pop_back();
        assert_eq!(m.size(), 1);
        assert_eq!(m.subrange(1).0.len(), 0);
        assert_eq!(m.subrange(0).0, &[1.0]);
        // Popping the only entry of column 0 empties it as well.
        m.pop_back();
        assert_eq!(m.size(), 0);
        assert_eq!(m.subrange(0).0.len(), 0);
        assert_eq!(m.subrange(1).0.len(), 0);
    }
}
