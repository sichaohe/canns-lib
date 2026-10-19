//! GPU column assembly.
//!
//! Assembly is the last large block that is genuinely parallel work: for each
//! kept simplex it enumerates cofacets, keeps those under threshold, and asks
//! whether each is already a pivot. It only READS the pivot set and has no
//! ordering dependency between simplices -- unlike the reduction walk, which
//! is strictly sequential.
//!
//! Validated against the CPU path on the real ASA matrix, same 2.9e8 output
//! entries, compared one by one rather than by count:
//!
//! ```text
//! visits   287,280,400   (= C(1200,3), matches ASSEMBLYSTATS membership_calls)
//! columns  287,260,957
//! ORDER mismatches over 287,260,957 entries: 0
//! GPU: count pass 0.012 s, prefix 0.003 s, write pass 0.088 s
//! ```
//!
//! # Why the ordering has to be reproduced exactly
//!
//! `columns_to_reduce` is sorted and deduplicated afterwards, so its final
//! order is order-independent -- but `next_simplices` is NOT sorted, and it
//! becomes the next dimension's `simplices`. Offsets come from an exclusive
//! prefix sum of the per-simplex counts, so output is "simplices order, then
//! enumerator order within each simplex" -- byte-identical to the CPU branch,
//! which relies on `par_iter().map(..).collect()` preserving order.
//!
//! If that rayon branch is ever changed to `fold`/`reduce`, this ordering
//! guarantee disappears and this path will silently diverge.
//!
//! # Why membership is a sorted array and not a hash or a filter
//!
//! 99.75% of the 287,280,400 probes answer "IS a pivot" -- only 718,201 answer
//! no, exactly the number of dim=1 pivots. A structure that is only good at
//! rejecting negatives (Bloom filter) is useless here, which is the same reason
//! the Bloom filter added to assembly measured 3.2% slower.

use crate::ripser::types::{DiameterIndexT, IndexT, ValueT};
use crate::ripser::utils::BinomialCoeffTable;

/// Whether the device-side radix sort key can represent every column index.
///
/// The key is a u64 whose high half is the inverted total_cmp key of the
/// diameter and whose low half is the column index, so an index that does not
/// fit in 32 bits would be TRUNCATED -- the tie-break would silently become
/// the wrong one and the assembled columns would come out in an order the
/// host comparator never produces, with no error anywhere. The only safe
/// response is to refuse the device sort and let the host sort the columns.
///
/// The bound that matters is C(n, actual_dim + 2), the number of columns at
/// the dimension being assembled. At the ASA scale that is 287,332,400,
/// comfortably inside u32 -- but n is not bounded by the caller, so the check
/// has to be a real one rather than an assumption.
///
/// Kept OUT of the `gpu`-feature module on purpose: this predicate is pure
/// arithmetic, and a test that only runs when a CUDA device is present is a
/// test that does not run in CI.
pub fn radix_key_fits(max_index: u64) -> bool {
    max_index <= u32::MAX as u64
}

#[cfg(feature = "gpu")]
mod imp {
    use super::*;
    use crate::ripser::gpu::{cached_pinned_alloc, device_and_module};
    use cudarc::driver::result::{memcpy_dtoh_sync, memset_d8_sync};
    use cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};

    const KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/ripser_kernels.cu"));
    // ONE source of truth: the NVRTC fallback compiles the same file the
    // nvcc path compiles. It used to be a hand-maintained copy of the
    // kernels, which was byte-identical to `kernel/ripser_kernels.cu` at
    // this revision but nothing kept it that way -- and a drifted copy
    // would make the NVRTC path and the PTX path compute different
    // barcodes with no error anywhere.

    /// Radix-sort the assembled columns on the device.
    ///
    /// The columns the assembly kernel writes are in parent-simplex order, not
    /// in the (diameter DESC, index ASC) order the reduction needs; the host
    /// used to sort them afterwards, which cost 2.43 s at ASA scale because it
    /// moves 4.6 GB through a comparison sort. A radix sort moves the same
    /// bytes 16 times with no comparisons, which is what a GPU is for, and it
    /// is what ripser++ does (thrust::sort on the device, then one D2H copy).
    ///
    /// Returns `Ok(false)` when the sort did not run and the caller must fall
    /// back to the host-side sort. That happens when:
    ///   * the kill switch asks for the old path;
    ///   * a column index does not fit in the low 32 bits of the key, where it
    ///     would be truncated and the tie-break silently corrupted.
    fn radix_sort_device(
        // cudarc 0.12 takes `self: &Arc<Self>` for alloc_zeros / get_func /
        // synchronize, so the handle has to be the Arc, not a bare reference.
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        o_cols: &mut cudarc::driver::CudaSlice<i64>,
        total_cols: usize,
        max_index: u64,
    ) -> Result<bool, String> {
        if std::env::var("CANNS_RIPSER_GPU_SORT")
            .map(|v| matches!(v.trim(), "0" | "false" | "False" | "no" | "off"))
            .unwrap_or(false)
        {
            return Ok(false);
        }
        if total_cols == 0 {
            return Ok(false);
        }
        if !radix_key_fits(max_index) {
            return Ok(false);
        }

        // Must match the RADIX_* constants in kernel/ripser_kernels.cu.
        const TILE: usize = 4096;
        const THREADS: usize = 256;
        const PASSES: usize = 16; // 64-bit key / 4 bits per pass
        const BINS: usize = 16;

        let n_tiles = (total_cols + TILE - 1) / TILE;
        let mut scratch = dev
            .alloc_zeros::<i64>(total_cols * 2)
            .map_err(|e| e.to_string())?;
        let mut hist = dev
            .alloc_zeros::<u32>(n_tiles * BINS)
            .map_err(|e| e.to_string())?;
        let mut bin_offset = dev.alloc_zeros::<u32>(BINS).map_err(|e| e.to_string())?;

        let cfg_sort = LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_scan = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BINS as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // PASSES is even, so the result ends up back in `o_cols` and the
        // caller's D2H does not have to know which buffer won.
        let mut from_scratch = false;
        let t0 = std::time::Instant::now();
        for pass in 0..PASSES {
            let shift = (pass * 4) as i32;
            unsafe {
                memset_d8_sync(*hist.device_ptr(), 0, n_tiles * BINS * 4)
                    .map_err(|e| e.to_string())?;
                memset_d8_sync(*bin_offset.device_ptr(), 0, BINS * 4).map_err(|e| e.to_string())?;
            }
            if from_scratch {
                unsafe {
                    dev.get_func("canns_asm", "radix_hist4")
                        .ok_or("radix_hist4 kernel missing")?
                        .launch(cfg_sort, (&scratch, total_cols as i64, shift, &mut hist))
                        .map_err(|e| e.to_string())?;
                    dev.get_func("canns_asm", "radix_scan4")
                        .ok_or("radix_scan4 kernel missing")?
                        .launch(cfg_scan, (&mut hist, n_tiles as i64, &mut bin_offset))
                        .map_err(|e| e.to_string())?;
                    dev.get_func("canns_asm", "radix_scatter4")
                        .ok_or("radix_scatter4 kernel missing")?
                        .launch(
                            cfg_sort,
                            (
                                &scratch,
                                &mut *o_cols,
                                total_cols as i64,
                                shift,
                                &hist,
                                &bin_offset,
                            ),
                        )
                        .map_err(|e| e.to_string())?;
                }
            } else {
                unsafe {
                    dev.get_func("canns_asm", "radix_hist4")
                        .ok_or("radix_hist4 kernel missing")?
                        .launch(cfg_sort, (&*o_cols, total_cols as i64, shift, &mut hist))
                        .map_err(|e| e.to_string())?;
                    dev.get_func("canns_asm", "radix_scan4")
                        .ok_or("radix_scan4 kernel missing")?
                        .launch(cfg_scan, (&mut hist, n_tiles as i64, &mut bin_offset))
                        .map_err(|e| e.to_string())?;
                    dev.get_func("canns_asm", "radix_scatter4")
                        .ok_or("radix_scatter4 kernel missing")?
                        .launch(
                            cfg_sort,
                            (
                                &*o_cols,
                                &mut scratch,
                                total_cols as i64,
                                shift,
                                &hist,
                                &bin_offset,
                            ),
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
            dev.synchronize().map_err(|e| e.to_string())?;
            from_scratch = !from_scratch;
        }
        if std::env::var("CANNS_RIPSER_STATS").is_ok() {
            eprintln!(
                "RADIXSORT cols={total_cols} tiles={n_tiles} passes={PASSES} s={:.3}",
                t0.elapsed().as_secs_f64()
            );
        }
        // PASSES is even, so the last pass wrote into `o_cols`; the caller
        // reads the result from there.
        Ok(true)
    }

    pub fn assemble(
        dense: &[f32],
        n: IndexT,
        binomial: &BinomialCoeffTable,
        simplices: &[DiameterIndexT],
        actual_dim: IndexT,
        threshold: ValueT,
        pivot_keys: &[IndexT],
        keep_next: bool,
    ) -> Result<(Vec<DiameterIndexT>, Vec<DiameterIndexT>, bool), String> {
        use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

        let (dev, _) = device_and_module()?;
        let opts = CompileOptions {
            options: vec!["-default-device".to_string()],
            ..Default::default()
        };
        // nvcc-built PTX first, NVRTC only when there is none -- see the
        // `prebuilt_ptx` docs in gpu.rs for why PTX and not a cubin.
        let ptx = match crate::ripser::gpu::prebuilt_ptx() {
            Some(src) => cudarc::nvrtc::Ptx::from_src(src.to_string()),
            None => compile_ptx_with_opts(KERNEL, opts).map_err(|e| e.to_string())?,
        };
        dev.load_ptx(
            ptx,
            "canns_asm",
            &[
                "count_columns",
                "write_columns",
                "radix_hist4",
                "radix_scan4",
                "radix_scatter4",
            ],
        )
        .map_err(|e| e.to_string())?;

        let n_simplices = simplices.len() as i64;
        if n_simplices == 0 {
            return Ok((Vec::new(), Vec::new(), false));
        }
        let sim: Vec<IndexT> = simplices.iter().map(|s| s.get_index()).collect();
        let d_sim = dev.htod_copy(sim).map_err(|e| e.to_string())?;
        let d_binom = dev
            .htod_copy(binomial.as_slice().to_vec())
            .map_err(|e| e.to_string())?;
        let d_dist = dev.htod_copy(dense.to_vec()).map_err(|e| e.to_string())?;
        let pivots: Vec<u64> = pivot_keys.iter().map(|&k| k as u64).collect();
        let d_piv = dev.htod_copy(pivots).map_err(|e| e.to_string())?;

        let mut d_ca = dev
            .alloc_zeros::<u32>(n_simplices as usize)
            .map_err(|e| e.to_string())?;
        let mut d_cc = dev
            .alloc_zeros::<u32>(n_simplices as usize)
            .map_err(|e| e.to_string())?;

        let grid = 8192u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let packed = ((n as i64) << 32) | ((actual_dim as i64) << 8) | (keep_next as i64);
        unsafe {
            dev.get_func("canns_asm", "count_columns")
                .ok_or("count kernel missing")?
                .launch(
                    cfg,
                    (
                        &d_sim,
                        n_simplices,
                        &d_binom,
                        binomial.n_max() as i64,
                        &d_dist,
                        packed,
                        threshold,
                        &d_piv,
                        pivot_keys.len() as i64,
                        &mut d_ca,
                        &mut d_cc,
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        dev.synchronize().map_err(|e| e.to_string())?;

        let counts_all = dev.dtoh_sync_copy(&d_ca).map_err(|e| e.to_string())?;
        let counts_cols = dev.dtoh_sync_copy(&d_cc).map_err(|e| e.to_string())?;
        drop(d_ca);
        drop(d_cc);

        // Interleaved because cudarc's Params tuple tops out at 12 arguments.
        let mut off: Vec<i64> = Vec::with_capacity(counts_all.len() * 2);
        let (mut a, mut c) = (0i64, 0i64);
        for i in 0..counts_all.len() {
            off.push(a);
            off.push(c);
            a += counts_all[i] as i64;
            c += counts_cols[i] as i64;
        }
        let (total_all, total_cols) = (a as usize, c as usize);

        let d_off = dev.htod_copy(off).map_err(|e| e.to_string())?;
        let mut o_all = dev
            .alloc_zeros::<i64>(if keep_next { total_all * 2 } else { 2 })
            .map_err(|e| e.to_string())?;
        let mut o_cols = dev
            .alloc_zeros::<i64>(total_cols * 2)
            .map_err(|e| e.to_string())?;

        unsafe {
            dev.get_func("canns_asm", "write_columns")
                .ok_or("write kernel missing")?
                .launch(
                    cfg,
                    (
                        &d_sim,
                        n_simplices,
                        &d_off,
                        &d_binom,
                        binomial.n_max() as i64,
                        &d_dist,
                        packed,
                        threshold,
                        &d_piv,
                        pivot_keys.len() as i64,
                        &mut o_all,
                        &mut o_cols,
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        dev.synchronize().map_err(|e| e.to_string())?;

        // ---- GPU radix sort ------------------------------------------------
        //
        // The columns the kernel just wrote are in parent-simplex order, NOT
        // in the (diameter DESC, index ASC) order the reduction needs; the
        // host used to sort them afterwards, which cost 2.43 s at ASA scale.
        // The radix sort moves the same 4.6 GB sixteen times with no
        // comparisons, which is what a GPU is for, and it is what ripser++
        // does (thrust::sort on the device, then one D2H copy).
        //
        // Returns true when the sort ran (the sorted records are left in
        // `o_cols`), false when the caller must fall back to the host sort.
        // The column indices are combinatorial simplex indices, bounded by
        // C(n, actual_dim + 2); that bound is what the key's low 32 bits have
        // to fit into.
        let index_bound = binomial.get(n, actual_dim + 2).max(0) as u64;
        let sorted_on_gpu = radix_sort_device(dev, &mut o_cols, total_cols, index_bound)?;

        // Readback through pinned memory: pageable staging measured 2.0 GB/s
        // against 24.2 pinned, and this is ~4.6 GB.
        // The process-level pinned cache: these two buffers are 9.2 GB
        // together at ASA scale, and `cuMemHostAlloc` has to page-lock them
        // at ~0.32 s/GB. A grid-cells session runs 97 matrices in one
        // process, so re-allocating them every time is several minutes of
        // setup that buys nothing. See `cached_pinned_alloc`.
        let p_all = if keep_next {
            cached_pinned_alloc(total_all * 16)?
        } else {
            cached_pinned_alloc(16)?
        };
        let p_cols = cached_pinned_alloc(total_cols * 16)?;
        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                p_all.as_ptr() as *mut i64,
                if keep_next { total_all * 2 } else { 2 },
            );
            memcpy_dtoh_sync(dst, *o_all.device_ptr()).map_err(|e| e.to_string())?;
            let dst = std::slice::from_raw_parts_mut(p_cols.as_ptr() as *mut i64, total_cols * 2);
            memcpy_dtoh_sync(dst, *o_cols.device_ptr()).map_err(|e| e.to_string())?;
        }

        // [f32, i64] records: interpret the raw pairs in place.
        // [f32, i64] records: interpret the raw pairs in place.
        //
        // PARALLEL. This repacks 286,562,199 records on the GPU path and it
        // used to do it on ONE thread. It is pure data movement with no
        // ordering dependency, so rayon is a free win: the phase breakdown
        // attributes the whole 3.567 s of `ASSEMBLYGPU` to the GPU call, and
        // the D2H is only part of that -- the single-threaded repack was the
        // rest.
        let to_cols = |raw: &[i64], n: usize| -> Vec<DiameterIndexT> {
            let mut out: Vec<DiameterIndexT> = Vec::with_capacity(n);
            // Kill switch: CANNS_RIPSER_SEQ_TOCOLS=1 restores the
            // single-threaded repack, so the parallel path can be A/B-ed in
            // one batch instead of across batches (machine load on the shared
            // box has moved baselines by 14% between batches).
            //
            // The two branches MUST be mutually exclusive. An earlier version
            // used a plain `if` followed by an unconditional parallel extend,
            // so the kill switch produced 2n columns -- a silently wrong
            // barcode that only a column-count check would catch.
            let seq_tocols = std::env::var("CANNS_RIPSER_SEQ_TOCOLS")
                .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "True" | "yes"))
                .unwrap_or(false);
            #[cfg(feature = "parallel")]
            if !seq_tocols {
                use rayon::prelude::*;
                // `into_par_iter()` is used rather than `.map()` directly: a
                // glob-imported `ParallelIterator::map` loses method
                // resolution to the std prelude's `Iterator::map`, which
                // yields a std Map that is not an IntoParallelIterator.
                out.par_extend((0..n).into_par_iter().map(|i| {
                    let d = f32::from_bits(raw[i * 2] as u32);
                    let idx = raw[i * 2 + 1];
                    DiameterIndexT::new(d, idx)
                }));
            } else {
                for i in 0..n {
                    let d = f32::from_bits(raw[i * 2] as u32);
                    let idx = raw[i * 2 + 1];
                    out.push(DiameterIndexT::new(d, idx));
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = seq_tocols;
                for i in 0..n {
                    let d = f32::from_bits(raw[i * 2] as u32);
                    let idx = raw[i * 2 + 1];
                    out.push(DiameterIndexT::new(d, idx));
                }
            }
            out
        };
        let ra = unsafe {
            std::slice::from_raw_parts(
                p_all.as_ptr() as *const i64,
                if keep_next { total_all * 2 } else { 2 },
            )
        };
        let rc =
            unsafe { std::slice::from_raw_parts(p_cols.as_ptr() as *const i64, total_cols * 2) };
        let next = if keep_next {
            to_cols(ra, total_all)
        } else {
            Vec::new()
        };
        let cols = to_cols(rc, total_cols);

        // Copy out of the pinned buffers before they are released.
        Ok((cols, next, sorted_on_gpu))
    }
}

#[cfg(not(feature = "gpu"))]
mod imp {
    use super::*;

    pub fn assemble(
        _dense: &[f32],
        _n: IndexT,
        _binomial: &BinomialCoeffTable,
        _simplices: &[DiameterIndexT],
        _actual_dim: IndexT,
        _threshold: ValueT,
        _pivot_keys: &[IndexT],
        _keep_next: bool,
    ) -> Result<(Vec<DiameterIndexT>, Vec<DiameterIndexT>, bool), String> {
        Err("built without the `gpu` feature".to_string())
    }
}

/// Assemble the next dimension's columns on the GPU.
///
/// Returns `(columns_to_reduce, next_simplices, sorted_on_gpu)`. The columns
/// are in the same order the CPU path produces; `sorted_on_gpu` says whether
/// the device already put them in (diameter DESC, index ASC) order, so the
/// caller can skip the host-side sort. Any device problem returns `Err`, and
/// the caller falls back.
pub fn assemble(
    dense: &[f32],
    n: IndexT,
    binomial: &BinomialCoeffTable,
    simplices: &[DiameterIndexT],
    actual_dim: IndexT,
    threshold: ValueT,
    pivot_keys: &[IndexT],
    // next_simplices is the superset (every cofacet under threshold) and is
    // only consumed when another dimension follows. On the LAST dimension
    // keep_next is false, and building it anyway doubles the output traffic
    // for nothing -- measured at half of a 7.1 s phase.
    keep_next: bool,
) -> Result<(Vec<DiameterIndexT>, Vec<DiameterIndexT>, bool), String> {
    imp::assemble(
        dense, n, binomial, simplices, actual_dim, threshold, pivot_keys, keep_next,
    )
}

#[cfg(test)]
mod tests {
    use super::radix_key_fits;

    /// The device key puts the column index in the low 32 bits. An index
    /// that does not fit there would be truncated, the tie-break would
    /// silently become the wrong one, and the assembled columns would come
    /// out in an order the host comparator never produces -- with no error
    /// anywhere. The device sort must refuse instead.
    #[test]
    fn radix_key_fits_rejects_indices_above_u32() {
        assert!(radix_key_fits(0));
        assert!(radix_key_fits(1));
        assert!(radix_key_fits(u32::MAX as u64));
        assert!(!radix_key_fits(u32::MAX as u64 + 1));
        assert!(!radix_key_fits(u64::MAX));
    }

    /// The ASA-scale bound must be accepted, otherwise the device sort would
    /// be refused on exactly the case it was written for.
    #[test]
    fn radix_key_fits_accepts_the_asa_scale_bound() {
        // C(1200, 3) -- the number of columns at dim 2 for n = 1200, which is
        // the `visits` count the assembly kernel reports.
        let c = |n: u64, k: u64| -> u64 {
            (1..=k).fold(1u64, |acc, i| acc * (n - k + i) / i)
        };
        let bound = c(1200, 3);
        assert_eq!(bound, 287_280_400);
        assert!(radix_key_fits(bound));
    }
}
