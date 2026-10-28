use crate::ripser::matrix::traits::{DistanceMatrix, EdgeProvider, HasCofacets, VertexBirth};
use crate::ripser::types::{
    CoefficientT, DiameterEntryT, DiameterIndexT, IndexT, PivotVal, ValueT,
};
use crate::ripser::utils::BinomialCoeffTable;
use rustc_hash::FxHashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Call counters for the assembly phase, gated behind `apparent-stats`.
/// Atomics rather than thread_local for the same reason as the pre-pass
/// counters: assembly runs on rayon workers while the STATS line prints on the
/// calling thread, so per-thread counters read back as zero.
///
/// These exist to decide one question: could the pivot map be replaced by a
/// SORTED ARRAY (ripser++'s design) without making assembly quadratic-ish?
/// `contains_key` on a sorted array is O(log n) with a cache miss per probe,
/// so the call count is what decides whether that trade is viable.
#[cfg(feature = "apparent-stats")]
pub mod assembly_stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static MEMBERSHIP: AtomicU64 = AtomicU64::new(0);
    pub static COFACET_VISITS: AtomicU64 = AtomicU64::new(0);
    #[inline]
    pub fn count_membership() {
        MEMBERSHIP.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn count_cofacet_visit() {
        COFACET_VISITS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "apparent-stats"))]
pub mod assembly_stats {
    #[inline(always)]
    pub fn count_membership() {}
    #[inline(always)]
    pub fn count_cofacet_visit() {}
}

/// One line per dimension when CANNS_RIPSER_STATS is set.
pub fn report_assembly_stats(dim: IndexT) {
    #[cfg(feature = "apparent-stats")]
    if std::env::var("CANNS_RIPSER_STATS").is_ok() {
        use std::sync::atomic::Ordering;
        eprintln!(
            "ASSEMBLYSTATS dim={dim} membership_calls={} cofacet_visits={}",
            assembly_stats::MEMBERSHIP.load(Ordering::Relaxed),
            assembly_stats::COFACET_VISITS.load(Ordering::Relaxed)
        );
    }
    #[cfg(not(feature = "apparent-stats"))]
    let _ = dim;
}

// Column assembly coordinator
pub struct ColumnAssembler<M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + HasCofacets + Sync,
{
    pub dist: M,
    pub n: IndexT,
    pub threshold: ValueT,
    pub binomial_coeff: BinomialCoeffTable,
    pub modulus: CoefficientT,
    pub verbose: bool,
}

impl<M> ColumnAssembler<M>
where
    M: DistanceMatrix + VertexBirth + EdgeProvider + HasCofacets + Sync,
{
    pub fn new(
        dist: M,
        n: IndexT,
        threshold: ValueT,
        binomial_coeff: BinomialCoeffTable,
        modulus: CoefficientT,
        verbose: bool,
    ) -> Self {
        Self {
            dist,
            n,
            threshold,
            binomial_coeff,
            modulus,
            verbose,
        }
    }

    /// Helper function to check for zero apparent pairs
    fn is_in_zero_apparent_pair(&self, cofacet: DiameterEntryT, dim: IndexT) -> bool {
        if dim > 0 {
            // Implement zero-apparent pair detection logic
            // For now, return false (disabled)
            false
        } else {
            false
        }
    }

    /// Keep the original enumerator arithmetic and threshold comparison.
    /// Clearing only removes reduction columns; it MUST NOT remove parent
    /// simplices needed for a subsequent dimension.
    #[allow(clippy::too_many_arguments)] // Explicit inputs keep shared and per-batch buffers separate.
    fn append_cofacets(
        &self,
        simplex: &DiameterIndexT,
        actual_dim: IndexT,
        dim: IndexT,
        keep_next_simplices: bool,
        pivot_column_index: &FxHashMap<IndexT, PivotVal>,
        columns: &mut Vec<DiameterIndexT>,
        next_simplices: &mut Vec<DiameterIndexT>,
    ) {
        let mut cofacets = self.dist.make_enumerator(
            DiameterEntryT::new(simplex.get_diameter(), simplex.get_index(), 1),
            actual_dim,
            self.n,
            &self.binomial_coeff,
            self.modulus,
            self.threshold,
        );
        while cofacets.has_next(false) {
            let cofacet = cofacets.next();
            assembly_stats::count_cofacet_visit();
            if cofacet.get_diameter() <= self.threshold {
                let idx = cofacet.get_index();
                let entry = DiameterIndexT::new(cofacet.get_diameter(), idx);
                if keep_next_simplices {
                    next_simplices.push(entry);
                }
                assembly_stats::count_membership();
                // NEGATIVE RESULT, kept so it is not retried: a 256 MB Bloom
                // filter in front of this measured 3.2% SLOWER (68.2 s -> 70.4 s
                // over two interleaved passes).
                //
                // The reason is a wrong assumption, not a bad filter. Each
                // dimension's assembly only ever sees THAT dimension's pivot
                // map: assembling dim=2 uses dim=1's map, which has 718,201
                // entries (~4.4 MB) and is entirely cache-resident. The 2.87e8
                // membership queries therefore never touch the 13.1 GB table,
                // so there is no DRAM round trip to hide and the filter is pure
                // overhead. A filter only pays where the probed structure
                // actually misses cache.
                let is_pivot = pivot_column_index.contains_key(&idx);
                if !is_pivot {
                    // The original parallel branch does not apply apparent
                    // pair filtering; preserve that feature-specific behavior.
                    #[cfg(feature = "parallel")]
                    columns.push(entry);
                    #[cfg(not(feature = "parallel"))]
                    if !self.is_in_zero_apparent_pair(cofacet, dim) {
                        columns.push(entry);
                    }
                }
            }
        }
        // `dim` is only consumed by the non-parallel apparent-pair stub.
        let _ = dim;
    }

    pub fn assemble_columns_to_reduce(
        &self,
        simplices: &mut Vec<DiameterIndexT>,
        columns_to_reduce: &mut Vec<DiameterIndexT>,
        pivot_column_index: &mut FxHashMap<IndexT, PivotVal>,
        dim: IndexT,
        dim_max: IndexT,
    ) {
        let actual_dim = dim - 1;
        columns_to_reduce.clear();

        // `dim` is the dimension being assembled. Its simplices are only
        // needed as parents if a later dimension will be assembled.
        let keep_next_simplices = dim < dim_max;
        let mut next_simplices = Vec::new();

        // CPU fallback paths, skipped when the GPU path succeeded.
        #[allow(unused_mut)]
        // True when the device already returned the columns in the order the
        // reduction needs, so the host-side sort can be skipped.
        let mut assembled_sorted = false;
        let mut assembled_on_gpu = {
            #[cfg(feature = "gpu")]
            {
                false
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        };

        // GPU path. Assembly is the one remaining large phase with no
        // ordering dependency between simplices, so it maps directly onto the
        // device. Validated on the real matrix: 287,260,957 output entries
        // compared one by one against this CPU branch, zero mismatches, GPU
        // side 0.102 s against ~12.3 s here.
        //
        // It is tried before the rayon branch and falls back on ANY device
        // problem -- correctness must never depend on having a GPU.
        #[cfg(feature = "gpu")]
        {
            // Only worth it at scale. The device round trip costs more than the
            // whole phase for small problems, and the regression fixtures are
            // n<=300. Expected cofacet visits is the quantity that decides.
            let expected_visits = simplices.len() as i64 * (self.n - actual_dim).max(1);
            if !simplices.is_empty() && expected_visits >= (1 << 20) {
                let t_prep = std::time::Instant::now();
                let mut pivots: Vec<IndexT> = pivot_column_index.keys().copied().collect();
                pivots.sort_unstable();
                let t_pivots = t_prep.elapsed().as_secs_f64();
                // `densify` transposes the lower triangle into a dense n*n
                // matrix, and the second write of each pair is STRIDED, so
                // every one of the n(n-1)/2 pairs touches a fresh cache line.
                // ripser++ does not do this: it uploads the packed lower
                // triangle (n*(n-1)/2 floats) and the kernel indexes it
                // directly. Timed separately because it is a prime suspect
                // for the assembly gap.
                let t_densify = std::time::Instant::now();
                let dense = crate::ripser::gpu::densify(&self.dist, self.n);
                let t_densify = t_densify.elapsed().as_secs_f64();
                let t0 = std::time::Instant::now();
                match crate::ripser::gpu_assembly::assemble(
                    &dense,
                    self.n,
                    &self.binomial_coeff,
                    simplices,
                    actual_dim,
                    self.threshold,
                    &pivots,
                    keep_next_simplices,
                ) {
                    Ok((cols, next, sorted_on_gpu)) => {
                        *columns_to_reduce = cols;
                        next_simplices = next;
                        // The device already put the columns in
                        // (diameter DESC, index ASC) order, so the host sort
                        // below would be a 2.43 s no-op.
                        assembled_sorted = sorted_on_gpu;
                        if std::env::var("CANNS_RIPSER_STATS").is_ok() {
                            eprintln!(
                                "ASSEMBLYGPU dim={dim} cols={} next={} s={:.3} \
                                 pivots_s={:.3} densify_s={:.3} gpu_s={:.3}",
                                columns_to_reduce.len(),
                                next_simplices.len(),
                                t0.elapsed().as_secs_f64(),
                                t_pivots,
                                t_densify,
                                t0.elapsed().as_secs_f64()
                            );
                        }
                        assembled_on_gpu = true;
                    }
                    Err(e) => {
                        if std::env::var("CANNS_RIPSER_STATS").is_ok() {
                            eprintln!("ASSEMBLYGPU dim={dim} unavailable: {e}");
                        }
                    }
                }
            }
        }

        if !assembled_on_gpu {
            #[cfg(feature = "parallel")]
            {
                if rayon::current_num_threads() == 1 {
                    // A one-thread Rayon pool still compiles the parallel branch.
                    // Stream directly instead of retaining per-parent vectors for
                    // the entire triangle/tetrahedron population.
                    for simplex in simplices.iter() {
                        self.append_cofacets(
                            simplex,
                            actual_dim,
                            dim,
                            keep_next_simplices,
                            pivot_column_index,
                            columns_to_reduce,
                            &mut next_simplices,
                        );
                    }
                } else {
                    // At most this many potential cofacets are represented by one
                    // batch (except one unavoidable parent when n exceeds the cap).
                    // Indexed parallel iteration plus ordered merge preserves the
                    // parent order used by the unmodified implementation.
                    let max_local_cofacets = 1_usize << 18;
                    let per_parent_bound =
                        (self.n as usize).saturating_sub(actual_dim as usize).max(1);
                    let batch_len = (max_local_cofacets / per_parent_bound).clamp(1, 4096);
                    for batch in simplices.chunks(batch_len) {
                        let results: Vec<(Vec<DiameterIndexT>, Vec<DiameterIndexT>)> = batch
                            .par_iter()
                            .map(|simplex| {
                                let mut local_columns = Vec::new();
                                let mut local_simplices = Vec::new();
                                self.append_cofacets(
                                    simplex,
                                    actual_dim,
                                    dim,
                                    keep_next_simplices,
                                    pivot_column_index,
                                    &mut local_columns,
                                    &mut local_simplices,
                                );
                                (local_columns, local_simplices)
                            })
                            .collect();
                        for (columns, next) in results {
                            columns_to_reduce.extend(columns);
                            next_simplices.extend(next);
                        }
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                for simplex in simplices.iter() {
                    self.append_cofacets(
                        simplex,
                        actual_dim,
                        dim,
                        keep_next_simplices,
                        pivot_column_index,
                        columns_to_reduce,
                        &mut next_simplices,
                    );
                }
            }
        }

        *simplices = next_simplices;

        // Sorting and deduplication.
        //
        // The SORT can be skipped when the GPU radix sort already produced
        // the exact same order -- see `radix_sort_device`. Sorting 286,562,199
        // records on the host cost 2.43 s; on the device it is a handful of
        // passes with no comparisons. Same comparator, same tie-break, so
        // this is a pure removal of work, not a change of semantics.
        //
        // ⚠️ The DEDUP CANNOT BE SKIPPED WITH IT. The kernel writes one entry
        // per (parent simplex, cofacet) pair, and a d-simplex is a cofacet of
        // d+1 different (d-1)-simplices, so the same column index is written
        // d+1 times -- always with the same diameter, because a cofacet's
        // diameter is a property of the cofacet, not of the parent it was
        // reached from. The host sort made those duplicates ADJACENT, which
        // is the precondition `dedup_by` needs. The radix sort orders by
        // (diameter DESC, index ASC), so they are still adjacent -- but the
        // removal itself still has to happen, or the reduction sees the same
        // simplex twice and pairs it twice.
        let t_sort = std::time::Instant::now();
        if !assembled_sorted {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                columns_to_reduce.par_sort_unstable_by(|a, b| {
                    b.get_diameter()
                        .total_cmp(&a.get_diameter())
                        .then_with(|| a.get_index().cmp(&b.get_index()))
                });
            }

            #[cfg(not(feature = "parallel"))]
            {
                columns_to_reduce.sort_unstable_by(|a, b| {
                    b.get_diameter()
                        .total_cmp(&a.get_diameter())
                        .then_with(|| a.get_index().cmp(&b.get_index()))
                });
            }
        }
        let t_sorted = t_sort.elapsed().as_secs_f64();

        // Differential test: the device order must be byte-for-byte the host
        // order. The final barcode md5 CANNOT catch a wrong order here --
        // the gate hashes each dimension's bars AFTER sorting them, so it is
        // blind to the assembly order. This compares the two orders element
        // by element instead, which is the check that actually constrains the
        // radix sort. Off by default: it clones the column vector twice.
        if assembled_sorted && std::env::var("CANNS_RIPSER_VERIFY_SORT").is_ok() {
            let t_v = std::time::Instant::now();
            // Host order: sort a copy, then dedup -- exactly what the
            // non-GPU path does.
            let mut host = columns_to_reduce.clone();
            host.sort_unstable_by(|a, b| {
                b.get_diameter()
                    .total_cmp(&a.get_diameter())
                    .then_with(|| a.get_index().cmp(&b.get_index()))
            });
            host.dedup_by(|a, b| a.get_index() == b.get_index());
            // Device order: dedup only -- the sort is what is under test.
            let mut dev = columns_to_reduce.clone();
            dev.dedup_by(|a, b| a.get_index() == b.get_index());
            let mut mismatch = 0usize;
            let mut first = String::new();
            for (i, (dv, hv)) in dev.iter().zip(host.iter()).enumerate() {
                if dv.get_diameter().to_bits() != hv.get_diameter().to_bits()
                    || dv.get_index() != hv.get_index()
                {
                    mismatch += 1;
                    if first.is_empty() {
                        first = format!(
                            "i={i} dev=(d={},i={}) host=(d={},i={})",
                            dv.get_diameter(),
                            dv.get_index(),
                            hv.get_diameter(),
                            hv.get_index()
                        );
                    }
                }
            }
            let len_mismatch = dev.len() != host.len();
            eprintln!(
                "SORTVERIFY dim={dim} gpu_n={} host_n={} order_mismatch={} \
                 len_mismatch={} first=[{}] s={:.3}",
                dev.len(),
                host.len(),
                mismatch,
                len_mismatch,
                first,
                t_v.elapsed().as_secs_f64()
            );
        }

        let t_dedup = std::time::Instant::now();
        columns_to_reduce.dedup_by(|a, b| a.get_index() == b.get_index());
        let t_deduped = t_dedup.elapsed().as_secs_f64();
        if std::env::var("CANNS_RIPSER_STATS").is_ok() {
            eprintln!(
                "ASSEMBLYSORT dim={dim} n={} sorted_on_gpu={} sort_s={:.3} dedup_s={:.3}",
                columns_to_reduce.len(),
                assembled_sorted,
                t_sorted,
                t_deduped
            );
        }

        // Columns assembly complete
        if self.verbose {
            eprintln!(
                "DEBUG: assemble dim={} complete, unique columns={}, unique next_simplices={}",
                dim,
                columns_to_reduce.len(),
                simplices.len()
            );
        }
    }
}

#[cfg(test)]
mod allocation_assembly_tests {
    use super::*;
    use crate::ripser::matrix::dense::CompressedDistanceMatrix;

    // Brute-force clique oracle independent of cofacet enumeration.
    fn cliques(
        matrix: &CompressedDistanceMatrix<true>,
        binomial: &BinomialCoeffTable,
        size: usize,
        threshold: f32,
    ) -> Vec<DiameterIndexT> {
        fn visit(
            matrix: &CompressedDistanceMatrix<true>,
            binomial: &BinomialCoeffTable,
            vertices: &mut Vec<usize>,
            size: usize,
            threshold: f32,
            out: &mut Vec<DiameterIndexT>,
        ) {
            if vertices.len() == size {
                let mut diameter = 0_f32;
                for i in 0..vertices.len() {
                    for j in 0..i {
                        diameter = diameter.max(matrix.get(vertices[i], vertices[j]));
                    }
                }
                if diameter <= threshold {
                    let index = vertices
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| binomial.get(v as i64, i as i64 + 1))
                        .sum();
                    out.push(DiameterIndexT::new(diameter, index));
                }
                return;
            }
            let first = vertices.last().map_or(0, |v| v + 1);
            for v in first..matrix.size() {
                vertices.push(v);
                visit(matrix, binomial, vertices, size, threshold, out);
                vertices.pop();
            }
        }
        let mut out = Vec::new();
        visit(matrix, binomial, &mut Vec::new(), size, threshold, &mut out);
        out
    }

    fn sort_entries(entries: &mut [DiameterIndexT]) {
        entries.sort_unstable_by(|a, b| {
            b.get_diameter()
                .total_cmp(&a.get_diameter())
                .then_with(|| a.get_index().cmp(&b.get_index()))
        });
    }

    fn check_assembly() {
        let n = 8_i64;
        for sparse in [false, true] {
            let mut distances = Vec::new();
            for i in 1..n {
                for j in 0..i {
                    distances.push(if sparse && (i + 2 * j) % 7 == 0 {
                        f32::INFINITY
                    } else {
                        ((i + j) % 3) as f32 // ties and zero edges
                    });
                }
            }
            let matrix = CompressedDistanceMatrix::<true>::from_distances(distances).unwrap();
            let binomial = BinomialCoeffTable::new(n, 6).unwrap();
            for threshold in [1_f32, 2_f32, f32::INFINITY] {
                for dim in 2_i64..=4 {
                    for dim_max in [dim, dim + 1] {
                        let mut expected = cliques(&matrix, &binomial, dim as usize + 1, threshold);
                        sort_entries(&mut expected);
                        // Existing pivots must disappear only from columns,
                        // never from the next dimension's parent simplices.
                        let mut pivots = FxHashMap::default();
                        for (i, entry) in expected.iter().enumerate() {
                            if i % 3 == 0 {
                                pivots.insert(entry.get_index(), PivotVal::new(i, 1));
                            }
                        }
                        let expected_columns: Vec<_> = expected
                            .iter()
                            .copied()
                            .filter(|entry| !pivots.contains_key(&entry.get_index()))
                            .collect();
                        let mut parents = cliques(&matrix, &binomial, dim as usize, threshold);
                        sort_entries(&mut parents);
                        let mut columns = vec![DiameterIndexT::new(99.0, -1)];
                        let assembler = ColumnAssembler::new(
                            &matrix,
                            n,
                            threshold,
                            binomial.clone(),
                            47,
                            false,
                        );
                        assembler.assemble_columns_to_reduce(
                            &mut parents,
                            &mut columns,
                            &mut pivots,
                            dim,
                            dim_max,
                        );
                        assert_eq!(
                            columns, expected_columns,
                            "columns dim={dim} max={dim_max} threshold={threshold}"
                        );
                        if dim < dim_max {
                            sort_entries(&mut parents);
                            assert_eq!(
                                parents, expected,
                                "parents dim={dim} max={dim_max} threshold={threshold}"
                            );
                        } else {
                            assert!(
                                parents.is_empty(),
                                "terminal dimension retained unused parents"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn assembly_matches_clique_oracle_and_terminal_storage() {
        #[cfg(feature = "parallel")]
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(check_assembly);
        }
        #[cfg(not(feature = "parallel"))]
        check_assembly();
    }

    fn check_multiple_batches() {
        // 8,128 edge parents exceed the 2,064-parent batch size at n=128.
        let n = 128_i64;
        let matrix = CompressedDistanceMatrix::<true>::from_distances(
            (1..n)
                .flat_map(|i| (0..i).map(move |j| ((i + 2 * j) % 4) as f32))
                .collect(),
        )
        .unwrap();
        let binomial = BinomialCoeffTable::new(n, 5).unwrap();
        let mut expected = cliques(&matrix, &binomial, 3, 3.0);
        sort_entries(&mut expected);
        assert_eq!(expected.len(), 341376);
        for dim_max in [2, 3] {
            let mut pivots = FxHashMap::default();
            for (i, entry) in expected.iter().enumerate() {
                if i % 5 == 0 {
                    pivots.insert(entry.get_index(), PivotVal::new(i, 1));
                }
            }
            let expected_columns: Vec<_> = expected
                .iter()
                .copied()
                .filter(|entry| !pivots.contains_key(&entry.get_index()))
                .collect();
            let mut parents = cliques(&matrix, &binomial, 2, 3.0);
            sort_entries(&mut parents);
            let mut columns = Vec::new();
            ColumnAssembler::new(&matrix, n, 3.0, binomial.clone(), 47, false)
                .assemble_columns_to_reduce(&mut parents, &mut columns, &mut pivots, 2, dim_max);
            assert_eq!(columns, expected_columns);
            if dim_max == 3 {
                sort_entries(&mut parents);
                assert_eq!(parents, expected);
            } else {
                assert!(parents.is_empty());
            }
        }
    }

    #[test]
    fn multiple_batches_match_clique_oracle() {
        #[cfg(feature = "parallel")]
        for threads in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(check_multiple_batches);
        }
        #[cfg(not(feature = "parallel"))]
        check_multiple_batches();
    }
}
