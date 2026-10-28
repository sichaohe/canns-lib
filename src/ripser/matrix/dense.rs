use crate::ripser::matrix::traits::{
    CofacetEnumerator, DistanceMatrix, EdgeProvider, HasCofacets, IndexableMatrix, VertexBirth,
};
use crate::ripser::types::{CoefficientT, DiameterEntryT, DiameterIndexT, IndexT, ValueT};
use crate::ripser::utils::{modp, BinomialCoeffTable};
use std::sync::OnceLock;

/// How the cofacet enumerator may stop reading distances as soon as the
/// running maximum is already above the threshold.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EarlyExitMode {
    /// The environment asked for it, whatever the data says.
    ForcedOn,
    /// The environment asked for the old behaviour.
    ForcedOff,
    /// The data decides.
    Gated,
}

/// The fraction of the pairs the threshold REJECTS, in millionths.
///
/// WHAT DECIDES THE EARLY EXIT. The early exit saves the distance reads of
/// the candidates the threshold rejects, and it costs one comparison per
/// read. So it pays when the rejection rate is high, and it LOSES when the
/// rejection rate is low -- measured on the 1,200-vertex scale-scan instance,
/// where the matrix is 5.76 MB and every read is a cache hit:
///
/// | threshold | early exit off | early exit on | |
/// |---|---|---|---|
/// | T (the maximum finite distance) | 12.63 s | 14.67 s | 1.16x, a loss |
/// | T/2 | 7.92 s | 10.48 s | 1.32x, a loss |
/// | T/4 | 1.15 s | 1.04 s | 0.90x, a win |
/// | T/10 | 0.65 s | 0.27 s | 0.41x, a win |
///
/// The same switch is 1.9x FASTER on `o3_4096`, where 98.7% of the
/// candidates are rejected and the matrix is 33.5 MB, so every read it skips
/// is a DRAM miss. The sign of the effect is therefore a property of the
/// DATA, not of the code.
///
/// The fraction is the complement of the fraction of pairs the threshold
/// keeps, which the algorithm layer knows exactly: it is the number of edges
/// in the filtration against the number of pairs. It is set once per call,
/// before the reduction starts.
static REJECTION_FRACTION: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The rejection rate above which the early exit pays for itself.
///
/// Measured on the benchmark host: the crossover sits between a rejection
/// rate of about 40% (where the early exit loses 32%) and about 80% (where
/// it wins 10%), on a matrix whose reads are cache hits. On a matrix whose
/// reads are DRAM misses the crossover is much lower, which is the case the
/// early exit was written for.
const EARLY_EXIT_REJECTION_GATE: u32 = 700_000;

/// The fraction of pairs the threshold rejects, as the gate sees it.
///
/// Reported in the STATS line, because the early exit's effect depends on it
/// and a table that does not say which way the gate went cannot be read.
pub fn rejection_fraction() -> f64 {
    REJECTION_FRACTION.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1_000_000.0
}

/// Whether the early exit is in force for this call.
pub fn early_exit_active() -> bool {
    early_exit_allowed()
}

/// Set the fraction of pairs the threshold rejects. Called by the algorithm
/// layer, once per call, before the reduction.
pub fn set_rejection_fraction(fraction: f64) {
    let millionths = if fraction.is_finite() && fraction > 0.0 {
        (fraction * 1_000_000.0).min(1_000_000.0) as u32
    } else {
        0
    };
    REJECTION_FRACTION.store(millionths, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the early exit may stop reading distances early.
///
/// `CANNS_RIPSER_COFACET_EARLY_EXIT=0` restores the old behaviour of paying
/// all `dim + 1` distance reads for every candidate; `=1` forces the early
/// exit on whatever the data says, which is what the A/B table measures.
/// With neither, the data decides.
fn early_exit_allowed() -> bool {
    static MODE: OnceLock<EarlyExitMode> = OnceLock::new();
    let mode = *MODE.get_or_init(|| match std::env::var("CANNS_RIPSER_COFACET_EARLY_EXIT") {
        Ok(v) => match v.trim() {
            "0" | "false" | "False" | "no" | "No" | "off" | "OFF" => EarlyExitMode::ForcedOff,
            "1" | "true" | "True" | "yes" | "Yes" | "on" | "ON" => EarlyExitMode::ForcedOn,
            _ => EarlyExitMode::Gated,
        },
        Err(_) => EarlyExitMode::Gated,
    });
    match mode {
        EarlyExitMode::ForcedOn => true,
        EarlyExitMode::ForcedOff => false,
        EarlyExitMode::Gated => {
            REJECTION_FRACTION.load(std::sync::atomic::Ordering::Relaxed)
                >= EARLY_EXIT_REJECTION_GATE
        }
    }
}

// Note: Removed ripser compatibility traits since we're eliminating ripser_old

#[derive(Debug, Clone)]
pub struct CompressedDistanceMatrix<const LOWER: bool> {
    pub(crate) distances: Vec<f32>,
    size: usize,
    // Optimize row_offsets only for lower triangular matrices (safer)
    row_offsets: Vec<usize>,
}

impl<const LOWER: bool> CompressedDistanceMatrix<LOWER> {
    pub fn convert_layout<const NEW_LOWER: bool>(&self) -> CompressedDistanceMatrix<NEW_LOWER> {
        if LOWER == NEW_LOWER {
            return CompressedDistanceMatrix {
                distances: self.distances.clone(),
                size: self.size,
                row_offsets: self.row_offsets.clone(),
            };
        }
        CompressedDistanceMatrix::<NEW_LOWER>::from_matrix(self)
    }

    pub fn from_distances(distances: Vec<f32>) -> Result<Self, String> {
        // Validate distances - reject NaN values
        for (i, &d) in distances.iter().enumerate() {
            if d.is_nan() {
                return Err(format!("NaN distance found at index {}", i));
            }
        }

        // compute the size of the matrix based on the number of distances
        // L = N * (N - 1) / 2  =>  8L + 1 = 4N^2 - 4N + 1 = (2N - 1)^2
        // => sqrt(8L + 1) = 2N - 1  =>  N = (sqrt(8L + 1) + 1) / 2
        let len = distances.len() as f64;
        let size_float = (1.0 + (1.0 + 8.0 * len).sqrt()) / 2.0;

        // make sure the size is an integer
        if size_float.fract() != 0.0 {
            return Err("Invalid number of distances for a compressed matrix.".to_string());
        }
        let size = size_float as usize;

        // Optimize row_offsets only for lower triangular matrices (safe)
        let mut row_offsets = Vec::with_capacity(size);
        for i in 0..size {
            if LOWER {
                // Lower triangular: row i contains i elements (j < i)
                // offset = 0 + 1 + 2 + ... + (i-1) = i*(i-1)/2
                // Row zero has no entries; avoid unsigned underflow in debug builds.
                row_offsets.push(i * i.saturating_sub(1) / 2);
            } else {
                // Upper triangular: use original calculation, no optimization
                row_offsets.push(0); // placeholder, not used
            }
        }

        Ok(Self {
            distances,
            size,
            row_offsets,
        })
    }

    /// return the size of the matrix
    pub fn size(&self) -> usize {
        self.size
    }

    /// get the distance between two indices
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }

        if LOWER {
            // Lower: i > j - use optimized row_offsets
            let (row, col) = if i > j { (i, j) } else { (j, i) };
            debug_assert!(row < self.row_offsets.len(), "Row {} out of bounds", row);
            debug_assert!(col < row, "Invalid column {} for row {}", col, row);
            let index = unsafe { *self.row_offsets.get_unchecked(row) + col };
            unsafe { *self.distances.get_unchecked(index) }
        } else {
            // Upper: i < j - keep original safe implementation
            let (row, col) = if i < j { (i, j) } else { (j, i) };
            // minus the number of edges in the rows after 'row'
            let total_edges = self.distances.len();
            let n = self.size;

            let tail_rows = n - 1 - row;
            let tail_edges = tail_rows * (tail_rows + 1) / 2;
            let index = total_edges - tail_edges + (col - row - 1);
            unsafe { *self.distances.get_unchecked(index) }
        }
    }

    /// Convert from any matrix that implements the `IndexableMatrix` trait.
    pub fn from_matrix<M: IndexableMatrix>(mat: &M) -> Self {
        let size = mat.size();
        if size <= 1 {
            return Self {
                distances: vec![],
                size,
                row_offsets: vec![],
            };
        }

        let num_distances = size * (size - 1) / 2;
        let mut distances = Vec::with_capacity(num_distances);

        // Fill the `distances` vector according to the layout
        if LOWER {
            // Lower triangular
            for i in 1..size {
                for j in 0..i {
                    distances.push(mat.get(i, j));
                }
            }
        } else {
            // Upper triangular
            for i in 0..size - 1 {
                for j in i + 1..size {
                    distances.push(mat.get(i, j));
                }
            }
        }

        // Calculate row_offsets
        let mut row_offsets = Vec::with_capacity(size);
        for i in 0..size {
            if LOWER {
                // Lower triangular: row i contains i elements (j < i)
                if i == 0 {
                    row_offsets.push(0);
                } else {
                    row_offsets.push(i * (i - 1) / 2);
                }
            } else {
                // Upper triangular: no optimization, use placeholder
                row_offsets.push(0);
            }
        }

        Self {
            distances,
            size,
            row_offsets,
        }
    }
}

pub type CompressedLowerDistanceMatrix = CompressedDistanceMatrix<true>;
pub type CompressedUpperDistanceMatrix = CompressedDistanceMatrix<false>;

impl<const LOWER: bool> IndexableMatrix for CompressedDistanceMatrix<LOWER> {
    fn size(&self) -> usize {
        self.size()
    }
    fn get(&self, i: usize, j: usize) -> f32 {
        self.get(i, j)
    }
}

impl<const LOWER: bool> DistanceMatrix for CompressedDistanceMatrix<LOWER> {
    fn size(&self) -> usize {
        CompressedDistanceMatrix::size(self)
    }

    fn get(&self, i: usize, j: usize) -> f32 {
        CompressedDistanceMatrix::get(self, i, j)
    }
}

impl<const LOWER: bool> VertexBirth for CompressedDistanceMatrix<LOWER> {
    fn vertex_birth(&self, _i: IndexT) -> ValueT {
        0.0
    }
}

impl<const LOWER: bool> EdgeProvider for CompressedDistanceMatrix<LOWER> {
    fn edges_under_threshold(
        &self,
        threshold: ValueT,
        n: IndexT,
        binom: &BinomialCoeffTable,
    ) -> Vec<DiameterIndexT> {
        let n_usize = n as usize;
        // Pre-allocate with estimated capacity to reduce reallocations
        let mut edges = Vec::with_capacity((n_usize * (n_usize - 1)) / 4);

        // Use row-by-row generation like C++ ripser to avoid O(n) vertex decoding
        for i in 1..n_usize {
            for j in 0..i {
                let length = self.get(i, j);
                if length <= threshold {
                    let idx = binom.get(i as IndexT, 2) + j as IndexT;
                    edges.push(DiameterIndexT::new(length, idx));
                }
            }
        }

        // Sort edges by diameter (ascending), then by index (descending) for H0 compatibility
        edges.sort_unstable_by(|a, b| {
            a.get_diameter()
                .total_cmp(&b.get_diameter())
                .then_with(|| b.get_index().cmp(&a.get_index()))
        });
        edges
    }
}

// Maximum simplex vertices stored inline in the coboundary enumerator. A
// simplex of homology dimension d has d+1 vertices; persistence beyond a
// handful of dimensions is computationally infeasible, so a small stack array
// removes the per-enumerator heap allocation that a Vec would incur.
const MAX_COFACET_SIMPLEX_VERTICES: usize = 16;

// Dense coboundary enumerator for compressed distance matrices
pub struct SimplexCoboundaryEnumerator<'a, M>
where
    M: DistanceMatrix,
{
    idx_below: IndexT,
    idx_above: IndexT,
    v: IndexT,
    k: IndexT,
    vertices: [IndexT; MAX_COFACET_SIMPLEX_VERTICES],
    num_vertices: usize,
    simplex: DiameterEntryT,
    modulus: CoefficientT,
    dist: &'a M,
    binomial_coeff: &'a BinomialCoeffTable,
    /// The value a distance has to reach for the cofacet to be rejected.
    ///
    /// ONE comparison per read, for both kinds of threshold:
    ///
    /// * a **finite** threshold `T`: the cofacet is rejected when a distance
    ///   exceeds `T`, and `d >= next_up(T)` is exactly `d > T` for every
    ///   finite `d` -- there is no float between `T` and `next_up(T)`. An
    ///   infinite distance is above `T` as well, so the same comparison
    ///   covers it.
    /// * an **infinite** threshold: nothing exceeds infinity, so the only way
    ///   a cofacet can be rejected is an infinite distance, and
    ///   `d >= INFINITY` is exactly `d == INFINITY`.
    ///
    /// The second case is why the rejection test cannot simply be `d > T`:
    /// with an infinite threshold that test is never true, so it would be pure
    /// overhead on every single read -- and the corpus runs at threshold =
    /// infinity, where the reads are cheap because the matrix is
    /// cache-resident. Measured on the 1,200-vertex ASA matrix, the
    /// two-comparison form made the run 7% SLOWER.
    reject_at: ValueT,
    /// Whether the early exit is enabled.
    ///
    /// The ablation arm turns it off, and it has to restore the OLD
    /// behaviour EXACTLY: a partly-on early exit is not an ablation, it is
    /// a third thing that no table describes.
    early_exit: bool,
}

impl<M> CofacetEnumerator for SimplexCoboundaryEnumerator<'_, M>
where
    M: DistanceMatrix,
{
    fn has_next(&self, all_cofacets: bool) -> bool {
        self.v >= self.k
            && (all_cofacets || self.binomial_coeff.get(self.v, self.k) > self.idx_below)
    }

    #[allow(clippy::should_implement_trait)]
    fn next(&mut self) -> DiameterEntryT {
        while self.binomial_coeff.get(self.v, self.k) <= self.idx_below {
            self.idx_below -= self.binomial_coeff.get(self.v, self.k);
            self.idx_above += self.binomial_coeff.get(self.v, self.k + 1);
            self.v -= 1;
            self.k -= 1;
            debug_assert!(self.k >= 0, "k should not be negative");
        }

        // Calculate cofacet diameter efficiently:
        // The diameter is max(original simplex diameter, max distance from new vertex to existing vertices)
        //
        // A row-view variant of this loop (borrowing row `v` once instead of
        // re-reading `row_offsets[v]` per element) was tried and measured
        // NEUTRAL: +0.04% at n=300, +1.6% at n=500. The compiler already
        // hoists it -- `row_offsets` is 9.6 KB and L1-resident -- so the loop
        // is not bound by that load. Not worth the extra trait surface.
        //
        // A SOFTWARE PREFETCH of the next few rows (this loop walks `v`
        // downward, so the upcoming rows are known) was also tried and came out
        // NEUTRAL: two interleaved passes disagreed in sign (+4.1%, -1.5%)
        // with a pre-pass difference of 1.3%, inside run-to-run noise. Do not
        // retry. Two reasons it cannot work here: `has_next()` depends only on
        // `v` and the binomial table, never on the distances just loaded, so
        // the out-of-order engine already overlaps one cofacet's misses with
        // the next; and prefetching 3 rows x 3 columns to serve 3 real loads is
        // a 3x instruction amplification that offsets any residual gain.
        // Latency in this loop is already covered by speculation.
        //
        // ⚠ BOTH of those experiments were run at n = 300..500, where the
        // whole distance matrix is a few hundred KB and every load is a
        // cache HIT. They therefore say nothing about the regime this loop
        // actually runs in for a real dataset: at o3_4096 the matrix is
        // 4096 points = 8.39M f32 = 33.5 MB, the rows are 16 KB apart, and
        // the reads miss. The fix that DOES scale with the matrix size is not
        // a better prefetch but FEWER READS -- see the early exit below.
        let mut cofacet_diameter = self.simplex.get_diameter();
        //
        // THE LOOP'S STATE IS HOISTED OUT OF IT. `self.dist.get(...)` is a
        // call through a shared reference, and the compiler cannot prove
        // that it does not write through `self`, so every field the loop
        // reads is reloaded from memory on every iteration -- and the loop
        // body then carries a chain of load, max, compare, branch, where the
        // branch waits for the max, which waits for the load. On a matrix
        // that fits in cache that chain is the bottleneck: the early exit
        // was measured 23% SLOWER on the 1,200-vertex scale-scan instance
        // (5.76 MB, cache-resident) while being 1.9x faster on o3_4096
        // (33.5 MB, every read a miss). A local cannot be aliased, so the
        // values are copied out once and the compiler keeps them in
        // registers.
        let reject_at = self.reject_at;
        let early_exit = self.early_exit;
        let v = self.v as usize;
        let vertices = &self.vertices[..self.num_vertices];
        for &w in vertices {
            let d = self.dist.get(v, w as usize);
            cofacet_diameter = cofacet_diameter.max(d);
            // EARLY EXIT.
            //
            // Every caller of this enumerator keeps a cofacet only when its
            // diameter is <= the threshold, and the true diameter is >= the
            // running maximum. So once the running maximum is already above
            // the threshold, the cofacet is rejected either way, and the
            // remaining distances cannot change that verdict.
            //
            // This is not a rare case. At o3_4096 (4096 points, threshold
            // 1.4) only 1.3% of the enumerated cofacet candidates are under
            // the threshold, and the old code paid all dim+1 distance reads
            // for every one of the other 98.7% -- each a scattered miss into
            // a 33.5 MB matrix.
            //
            // THE COMPARISON IS ON THE DISTANCE, NOT ON THE RUNNING MAXIMUM.
            // The two are the same test: the running maximum can only reach
            // `reject_at` through a READ, because the simplex's own diameter
            // is <= the threshold and `reject_at` is the next float above it
            // (or infinity, which no finite simplex diameter reaches). So
            // `d >= reject_at` fires exactly when the running maximum would
            // have -- and it is ONE LINK SHORTER on the critical path: the
            // branch waits for the load, not for the load and the max. The
            // max is still computed, because the value returned is the
            // partial maximum and every caller filters on the threshold
            // before it uses the diameter.
            //
            if early_exit && d >= reject_at {
                // ONE comparison, covering both rejections: a distance above
                // a finite threshold, and an infinite distance under an
                // infinite threshold. The diameter returned in that case is
                // a partial maximum, but it is already above the threshold,
                // so no caller can observe it: a caller that kept it would
                // keep the exact diameter too.
                break;
            }
        }

        let cofacet_index =
            self.idx_above + self.binomial_coeff.get(self.v, self.k + 1) + self.idx_below;
        self.v -= 1;

        // `sign * base mod modulus`. Two divisions avoided, both provably
        // behaviour-preserving: `modp` on an already-reduced value is the
        // identity, and when `base == 1` (the initial coboundary of a simplex
        // always has coefficient 1) `sign` is already in [0, modulus).
        let sign: i32 = if self.k & 1 != 0 {
            (self.modulus - 1) as i32
        } else {
            1
        };
        let base = self.simplex.get_coefficient() as i32;
        let cofacet_coefficient: CoefficientT = if base == 1 {
            sign as CoefficientT
        } else {
            modp(
                ((sign * base) % (self.modulus as i32)) as CoefficientT,
                self.modulus,
            )
        };

        DiameterEntryT::new(cofacet_diameter, cofacet_index, cofacet_coefficient)
    }
}

impl<'a, M> SimplexCoboundaryEnumerator<'a, M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + Sync,
{
    pub fn new(
        simplex: DiameterEntryT,
        dim: IndexT,
        n: IndexT,
        dist: &'a M,
        binomial_coeff: &'a BinomialCoeffTable,
        modulus: CoefficientT,
        threshold: ValueT,
    ) -> Self {
        // Fill the simplex vertices into an inline stack buffer (no heap alloc).
        let mut vertices = [0 as IndexT; MAX_COFACET_SIMPLEX_VERTICES];
        let num_vertices =
            fill_simplex_vertices(simplex.get_index(), dim, n, binomial_coeff, &mut vertices);

        Self {
            idx_below: simplex.get_index(),
            idx_above: 0,
            v: n - 1,
            k: dim + 1,
            vertices,
            num_vertices,
            simplex,
            modulus,
            dist,
            binomial_coeff,
            reject_at: if threshold.is_finite() {
                // The next float above the threshold: `d >= reject_at` is
                // then exactly `d > threshold`.
                f32::from_bits(threshold.to_bits() + 1)
            } else {
                // An infinite threshold rejects nothing except an infinite
                // distance, and `d >= INFINITY` is exactly `d == INFINITY`.
                f32::INFINITY
            },
            early_exit: early_exit_allowed(),
        }
    }
}

// Extract simplex vertices into a caller-provided buffer, returning the count.
// Avoids the heap allocation of returning a Vec on the hot enumeration path.
fn fill_simplex_vertices(
    mut idx: IndexT,
    dim: IndexT,
    mut n: IndexT,
    binomial_coeff: &BinomialCoeffTable,
    out: &mut [IndexT; MAX_COFACET_SIMPLEX_VERTICES],
) -> usize {
    let count = (dim + 1) as usize;
    debug_assert!(
        count <= MAX_COFACET_SIMPLEX_VERTICES,
        "simplex dimension {} exceeds inline vertex buffer",
        dim
    );
    n -= 1;

    // Diameter uses max over vertices, so ordering is irrelevant; write directly.
    for (slot, k) in out.iter_mut().zip((1..=dim + 1).rev()) {
        n = get_max_vertex(idx, k, n, binomial_coeff);
        *slot = n;
        idx -= binomial_coeff.get(n, k);
    }
    count
}

#[inline(always)]
pub fn get_max_vertex(
    idx: IndexT,
    k: IndexT,
    n: IndexT,
    binomial_coeff: &BinomialCoeffTable,
) -> IndexT {
    // Fast path for k=2 using closed-form solution (most common case)
    if k == 2 {
        // For k=2: binom(n, 2) = n*(n-1)/2 = idx
        // => n^2 - n - 2*idx = 0
        // => n = (1 + sqrt(1 + 8*idx)) / 2
        let discriminant = 1.0 + 8.0 * (idx as f64);
        let n_float = (1.0 + discriminant.sqrt()) / 2.0;
        return n_float.floor() as IndexT;
    }

    // General case: binary search
    let mut top = n;
    let bottom = k - 1;

    let binom_top = binomial_coeff.get(top, k);

    if binom_top > idx {
        let mut count = top - bottom;

        while count > 0 {
            let step = count >> 1;
            let mid = top - step;

            let binom_mid = binomial_coeff.get(mid, k);

            if binom_mid > idx {
                top = mid - 1;
                count -= step + 1;
            } else {
                count = step;
            }
        }
    }

    top
}

impl<const LOWER: bool> HasCofacets for CompressedDistanceMatrix<LOWER> {
    type Enumerator<'a> = SimplexCoboundaryEnumerator<'a, Self>;

    fn cofacet_enumerator<'a>(
        &'a self,
        simplex: DiameterEntryT,
        dim: IndexT,
        n: IndexT,
        binomial_coeff: &'a BinomialCoeffTable,
        modulus: CoefficientT,
        threshold: ValueT,
    ) -> Self::Enumerator<'a> {
        SimplexCoboundaryEnumerator::new(simplex, dim, n, self, binomial_coeff, modulus, threshold)
    }
}

// Note: Removed ripser compatibility traits - no longer needed

#[cfg(test)]
mod tests {
    use super::CompressedDistanceMatrix;

    #[test]
    fn lower_triangle_offsets_include_empty_first_row() {
        let single = CompressedDistanceMatrix::<true>::from_distances(vec![]).unwrap();
        assert_eq!(single.size(), 1);
        assert_eq!(single.row_offsets, [0]);
        assert_eq!(single.get(0, 0), 0.0);

        let matrix =
            CompressedDistanceMatrix::<true>::from_distances(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        assert_eq!(matrix.size(), 4);
        assert_eq!(matrix.row_offsets, [0, 0, 1, 3]);
        let mut index = 0;
        for i in 0..matrix.size() {
            assert_eq!(matrix.get(i, i), 0.0);
            for j in 0..i {
                let expected = (index + 1) as f32;
                assert_eq!(matrix.get(i, j), expected);
                assert_eq!(matrix.get(j, i), expected);
                index += 1;
            }
        }
    }
}
