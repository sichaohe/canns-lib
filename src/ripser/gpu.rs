//! GPU apparent-pair detection.
//!
//! `get_zero_apparent_cofacet` decides the fate of 99.89% of columns on the ASA
//! matrix (286,263,529 of 286,562,199), and it is a pure function of
//! (column, dim): it reads only the distance matrix, the binomial table, `n`
//! and the threshold, never the pivot map. That makes it exactly the shape of
//! work a GPU is good at, and exactly the shape of work that can be moved
//! without touching the sequential walk that follows.
//!
//! Measured, same 2.9e8 columns on an A100-SXM4-80GB:
//!
//! ```text
//! CPU (rayon, 16 threads) : 12.0 - 15.5 s
//! GPU kernel              :  0.093 s          129 - 167x
//! pinned readback         :  0.095 s          (24.2 GB/s; pageable is 2.0)
//! ```
//!
//! The kernel is a transcription of `MatrixReducer::get_zero_apparent_cofacet`
//! and must stay in lockstep with it -- same enumerator state machine, same
//! early-exit order, same threshold filter before the equality test. It is
//! checked bitwise against the CPU path by the 224-cell regression grid.
//!
//! Everything here is behind the opt-in `gpu` cargo feature, and every entry
//! point degrades to `None` rather than failing: a build without CUDA, a
//! machine without a device, or a driver hiccup all fall back to the existing
//! CPU pre-pass. The library never *requires* a GPU to be correct.

use crate::ripser::matrix::traits::DistanceMatrix;
use crate::ripser::types::{CoefficientT, DiameterEntryT, DiameterIndexT, IndexT, ValueT};
use crate::ripser::utils::BinomialCoeffTable;

/// One column's result. 16 bytes, naturally aligned, so the readback is a
/// single `memcpy` instead of three strided ones.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApparentResult {
    /// Apparent cofacet's combinatorial index, or -1.
    pub index: IndexT,
    /// Apparent cofacet's diameter (ignored when `index == -1`).
    pub death: ValueT,
    /// Coefficient the enumerator assigns to that cofacet.
    pub coeff: CoefficientT,
    /// Padding, kept so the struct stays 16 bytes.
    pub _pad: CoefficientT,
}

impl ApparentResult {
    fn sentinel() -> Self {
        Self {
            index: -1,
            death: -1.0,
            coeff: 0,
            _pad: 0,
        }
    }
}

/// Below this many columns the GPU pre-pass wins outright and the transfer
/// costs more than it saves. Measured crossover is well below this; the
/// threshold is deliberately conservative because the GPU path allocates the
/// full result array up front (16 B/column) where the CPU path streams it.
pub const GPU_MIN_COLUMNS: usize = 1 << 16;

/// The device initialisation is paid ONCE per process, and it is not small:
/// the CUDA context and the module load cost ~0.245 s on the benchmark host.
/// The GPU pre-pass saves about 60 ns per column -- the CPU pre-pass against
/// the kernel, measured on the grid-cell corpus, where the GPU arm is 40 ms
/// faster on a 500K-column instance.
///
/// So the device only pays for itself at about four million columns, and
/// below that the GPU is a net LOSS, and the loss is per PROCESS: a
/// 500K-column instance saves 40 ms of pre-pass and pays 245 ms of context.
/// That is why the grid-cell corpus, whose matrices run at dimension 1, was
/// 25% SLOWER on the GPU build than on the CPU build -- the device was
/// bought for a saving it could not repay.
pub const GPU_INIT_BREAK_EVEN_COLUMNS: usize = 1 << 22;

/// The columns this process has offered the GPU pre-pass.
///
/// The device initialisation is a per-PROCESS cost, so the saving has to be
/// counted the same way: a process that makes many calls on small instances
/// earns the device, and one that makes a single call on a small instance
/// does not pay for it.
static GPU_COLUMNS_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the GPU pre-pass is worth the device, for a call with this many
/// columns.
///
/// The context is paid once. If it is already paid, the only question left
/// is whether the transfer pays for itself -- the old threshold's question.
/// If it is not, the process has to have earned it: the columns it has
/// offered so far, plus these, have to be worth the ~0.245 s the device
/// costs.
pub fn worth_the_device(columns: usize) -> bool {
    use std::sync::atomic::Ordering;
    let seen = GPU_COLUMNS_SEEN.fetch_add(columns as u64, Ordering::Relaxed) + columns as u64;
    available() || seen >= GPU_INIT_BREAK_EVEN_COLUMNS as u64
}

#[cfg(feature = "gpu")]
/// See `imp::prebuilt_ptx` -- nvcc-built PTX from `kernel/ripser_kernels.cu`.
pub(crate) use imp::prebuilt_ptx;
#[cfg(feature = "gpu")]
mod imp {
    use super::*;
    use std::os::raw::c_void;

    use cudarc::driver::result::memcpy_dtoh_sync;
    use cudarc::driver::sys::{self, CU_MEMHOSTALLOC_PORTABLE};
    use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    use libloading::Library;
    use std::sync::OnceLock;

    /// One device and one module for the whole process.
    ///
    /// At module level rather than inside `device()`, so that asking whether
    /// the device is usable can look at it WITHOUT creating it.
    static DEV: OnceLock<
        Result<(std::sync::Arc<CudaDevice>, cudarc::driver::CudaFunction), String>,
    > = OnceLock::new();

    const KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/ripser_kernels.cu"));
    // ONE source of truth: the NVRTC fallback compiles the same file the
    // nvcc path compiles. It used to be a hand-maintained copy of the
    // kernels, which was byte-identical to `kernel/ripser_kernels.cu` at
    // this revision but nothing kept it that way -- and a drifted copy
    // would make the NVRTC path and the PTX path compute different
    // barcodes with no error anywhere.

    type HostAllocFn = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> sys::CUresult;
    type HostFreeFn = unsafe extern "C" fn(*mut c_void) -> sys::CUresult;

    /// cudarc binds `cuMemHostAlloc` but loads libcuda with dlopen and keeps
    /// the handle private, so a plain `extern "C"` fails at LINK time even
    /// though the symbol is in the process. Resolve it the same way cudarc does.
    struct PinnedApi {
        _lib: Library,
        alloc: HostAllocFn,
        free: HostFreeFn,
    }

    impl PinnedApi {
        fn open() -> Result<Self, Box<dyn std::error::Error>> {
            unsafe {
                let lib = Library::new("libcuda.so.1")?;
                // Copy the fn pointers out before `lib` moves: Symbol borrows it.
                let alloc = *lib.get::<HostAllocFn>(b"cuMemHostAlloc\0")?;
                let free = *lib.get::<HostFreeFn>(b"cuMemFreeHost\0")?;
                Ok(Self {
                    _lib: lib,
                    alloc,
                    free,
                })
            }
        }
    }

    struct Pinned {
        ptr: *mut c_void,
        len: usize,
    }

    impl Pinned {
        fn alloc(api: &PinnedApi, len: usize) -> Result<Self, String> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: valid out-pointer, non-zero size, PORTABLE flag.
            let rc = unsafe { (api.alloc)(&mut ptr, len, CU_MEMHOSTALLOC_PORTABLE) };
            if rc != sys::CUresult::CUDA_SUCCESS || ptr.is_null() {
                return Err(format!("cuMemHostAlloc failed: {rc:?}"));
            }
            Ok(Self { ptr, len })
        }

        fn as_slice<T>(&self, count: usize) -> &[T] {
            debug_assert_eq!(std::mem::size_of::<T>() * count, self.len);
            // SAFETY: buffer is `len` bytes and was written by an exact-length copy.
            unsafe { std::slice::from_raw_parts(self.ptr as *const T, count) }
        }
    }

    impl Drop for Pinned {
        fn drop(&mut self) {
            // SAFETY: freed through the same API that allocated it; the Library
            // handle is kept alive in the caller's OnceLock.
            unsafe {
                if let Some(api) = pinned_api() {
                    (api.free)(self.ptr);
                }
            }
        }
    }

    /// PTX produced from `kernel/ripser_kernels.cu` by nvcc in `build.rs`.
    ///
    /// WHY A .cu FILE. The kernels used to be Rust string literals handed to
    /// `cudarc::nvrtc::compile_ptx_with_opts` at runtime, which constrains the
    /// source: NVRTC ships no system headers (so `<cstdint>` is unavailable and
    /// `i64` had to be hand-declared), there is no `__forceinline__`, and
    /// `cudarc::safe::Params` is a tuple capped at 12 arguments
    /// (`impl_launch!` stops at `[A..L]`). nvcc removes all three, which is what
    /// lets ripser++'s kernels be translated closely instead of paraphrased.
    ///
    /// WHY PTX AND NOT A CUBIN. build.rs emits both, because they fail differently:
    /// a cubin is loaded with no JIT and is therefore independent of the driver's
    /// CUDA version, while PTX is JIT-ed at load time and is rejected outright when
    /// a newer toolkit produced it (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` -- the
    /// flashbrain2 A6000 case: driver 550 / CUDA 12.4, toolkit 12.5).
    /// But cudarc 0.12 exposes no way to feed raw bytes: `Ptx` is built from a
    /// file (`cuModuleLoad`, cubin only) or from a source string
    /// (`cuModuleLoadData`, PTX text only -- `CString::new` rejects the interior
    /// NULs a cubin is full of), and `CudaFunction` has no public constructor, so
    /// the raw `result::module::load_data` path is unreachable from outside the
    /// crate. **A shippable crate therefore cannot carry the cubin here**, and PTX
    /// -- exactly the JIT exposure the NVRTC path already had -- is what actually
    /// loads. The cubin is still emitted so a same-machine dev run can use it, and
    /// so the day cudarc grows a bytes constructor nothing else has to change.
    ///
    /// Returns `None` when no nvcc ran at build time, and the caller falls back to
    /// the NVRTC string. That is deliberate: building with the `gpu` feature must
    /// not require nvcc.
    pub(crate) fn prebuilt_ptx() -> Option<&'static str> {
        static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ripser_kernels.ptx"));
        if PTX.trim().is_empty() {
            None
        } else {
            Some(PTX)
        }
    }

    fn pinned_api() -> Option<&'static PinnedApi> {
        static API: OnceLock<Option<PinnedApi>> = OnceLock::new();
        API.get_or_init(|| {
            // Best effort: a host without libcuda simply gets the CPU path.
            std::panic::catch_unwind(PinnedApi::open)
                .ok()
                .and_then(|r| r.ok())
        })
        .as_ref()
    }

    /// One device and one module for the process; re-initialising per call
    /// would cost more than the kernel.
    pub(crate) fn device(
    ) -> Result<&'static (std::sync::Arc<CudaDevice>, cudarc::driver::CudaFunction), String> {
        DEV.get_or_init(|| {
            let r = init_device();
            if let Err(e) = &r {
                // The error is RECORDED. It used to be returned to whoever
                // asked, and whoever asked was the reduction, which asked on
                // every call to fill in one statistics field -- so asking
                // created the device, and every call of a GPU build paid for
                // a CUDA context (~0.25 s on this host) whether or not it
                // ever touched the device.
                if let Ok(mut slot) = init_error().lock() {
                    *slot = Some(e.clone());
                }
            }
            r
        })
        .as_ref()
        .map_err(|e| e.clone())
    }

    /// The device and the kernel, initialised. Every step is named: a
    /// swallowed `.ok()?` chain makes a silent fallback indistinguishable
    /// from "the GPU was never tried".
    fn init_device(
    ) -> Result<(std::sync::Arc<CudaDevice>, cudarc::driver::CudaFunction), String> {
        if pinned_api().is_none() {
            return Err("libcuda.so.1 could not be dlopen'd".to_string());
        }
        let dev = CudaDevice::new(0).map_err(|e| format!("device 0: {e}"))?;
        // nvcc-built PTX first, NVRTC only when there is none.
        let ptx = match prebuilt_ptx() {
            Some(src) => cudarc::nvrtc::Ptx::from_src(src.to_string()),
            None => {
                let opts = CompileOptions {
                    options: vec!["-default-device".to_string()],
                    ..Default::default()
                };
                compile_ptx_with_opts(KERNEL, opts)
                    .map_err(|e| format!("nvrtc compile failed: {e}"))?
            }
        };
        dev.load_ptx(ptx, "canns", &["detect_apparent"])
            .map_err(|e| format!("load_ptx: {e}"))?;
        let f = dev
            .get_func("canns", "detect_apparent")
            .ok_or("get_func returned None")?;
        Ok((dev, f))
    }

    /// The recorded initialisation error, if the device could not be set up.
    fn init_error() -> &'static std::sync::Mutex<Option<String>> {
        static SLOT: OnceLock<std::sync::Mutex<Option<String>>> = OnceLock::new();
        SLOT.get_or_init(|| std::sync::Mutex::new(None))
    }

    /// Whether the device is usable -- WITHOUT CREATING IT.
    ///
    /// "Is there a GPU here" is a question, and the answer must not be the
    /// thing that pays for the GPU: the CUDA context costs ~0.25 s on this
    /// host. The device is usable when it has been initialised and worked;
    /// before that, the honest answer is "not tried yet".
    pub fn available() -> bool {
        matches!(DEV.get(), Some(Ok(_)))
    }

    /// Why the GPU is unusable, for the STATS line. Empty when it works.
    ///
    /// THE RECORDED ERROR, NOT A PROBE. This used to call `device()`, which
    /// created the CUDA context, and the reduction called it on every call to
    /// fill in one statistics field. So every call of a GPU build paid
    /// ~0.25 s for a context it might never use -- including the calls whose
    /// work was far too small to be worth the device at all. Now the error is
    /// recorded when the device is initialised, and asking for it is free.
    pub fn last_error() -> String {
        init_error()
            .lock()
            .ok()
            .and_then(|s| s.clone())
            .unwrap_or_default()
    }

    pub fn detect(
        dense: &[f32],
        n: IndexT,
        n_max: usize,
        binomial: &[IndexT],
        cols: Vec<IndexT>,
        dim: IndexT,
        threshold: ValueT,
        modulus: CoefficientT,
        ratio: f32,
    ) -> Result<ApparentRecords, String> {
        // Phase timers. Compiled with `apparent-stats` or `gpu-phase-stats`;
        // zero cost otherwise. They exist because THREE "obvious" wrapper
        // optimisations all measured slower (pinned H2D +13%, single fused
        // output array +5%, Bloom filter +3.2%), which says the itemised cost
        // model behind them was wrong. Measure before optimising again.
        //
        // `gpu-phase-stats` exists separately because `apparent-stats` also
        // enables the cofacet counters, which inflate the walk itself from
        // 25.5 s to 79.1 s -- a kernel time read under that feature is a
        // measurement of the counters, not of the kernel.
        macro_rules! phase {
            ($t:expr, $label:literal) => {{
                #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
                {
                    $t = Some(std::time::Instant::now());
                    let _ = &$label;
                }
                #[cfg(not(any(feature = "apparent-stats", feature = "gpu-phase-stats")))]
                let _ = &$label;
            }};
        }
        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        let (
            mut t_dense,
            mut t_alloc,
            mut t_upload,
            mut t_kernel,
            mut t_pin,
            mut t_read,
            mut t_build,
        ): (
            Option<std::time::Instant>,
            Option<std::time::Instant>,
            Option<std::time::Instant>,
            Option<std::time::Instant>,
            Option<std::time::Instant>,
            Option<std::time::Instant>,
            Option<std::time::Instant>,
        ) = (None, None, None, None, None, None, None);
        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        let stamp = |slot: &mut Option<std::time::Instant>| {
            if let Some(prev) = slot.replace(std::time::Instant::now()) {
                let _ = prev.elapsed();
            }
        };

        let (dev, _) = device().map_err(|e| e)?;
        let n_cols = cols.len() as i64;
        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        let t_dev = std::time::Instant::now();
        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        let t0 = std::time::Instant::now();

        let d_dev = dev.htod_copy(dense.to_vec()).map_err(|e| e.to_string())?;
        let b_dev = dev
            .htod_copy(binomial.to_vec())
            .map_err(|e| e.to_string())?;
        // NEGATIVE RESULT: pinning this upload was tried and reverted. Staging
        // 2.3 GiB through page-locked memory looked obviously right (pageable
        // runs at 2.0 GB/s, pinned at 24.2), but it measured 6.0 s -> 6.8 s
        // overall. The reason is that `cuMemHostAlloc` itself costs ~0.73 s
        // for this size, plus ~0.3 s to copy the columns in: ~1.1 s total
        // versus ~1.0 s pageable. Pinning only pays for REPEATED transfers over
        // a buffer that is reused; a one-shot upload of a freshly built array
        // cannot amortise the allocation.
        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        let t_cols_upload = std::time::Instant::now();
        // The Vec is MOVED into the device slice: cudarc keeps it as the
        // slice's host buffer and transfers from it in place. The old
        // `cols.to_vec()` copied all 807 MB first -- 0.23 s of pure memcpy
        // for a buffer the driver was going to read anyway.
        let n_cols_host = cols.len();
        let c_dev = dev.htod_copy(cols).map_err(|e| e.to_string())?;
        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        stamp(&mut t_upload);

        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        stamp(&mut t_alloc);
        // One 16-byte record per column, matching `DiameterEntryT`.
        // Allocated as i64 pairs rather than as a Rust struct: cudarc's
        // allocator only accepts the POD types it knows, and the 16-byte
        // record is exactly two of them. The byte layout is unchanged.
        let mut o_rec = dev
            .alloc_zeros::<i64>(n_cols_host * 2)
            .map_err(|e| e.to_string())?;
        // Two bits per column: is-apparent and produces-a-bar.
        let flag_words = (n_cols_host + 31) / 32;
        let mut o_flags = dev
            .alloc_zeros::<u64>(flag_words)
            .map_err(|e| e.to_string())?;

        let grid = ((n_cols_host + 255) / 256).min(8192) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: every pointer is a live allocation of at least `cols.len()`
        // elements and the kernel writes exactly that many.
        unsafe {
            dev.get_func("canns", "detect_apparent")
                .ok_or("kernel missing")?
                .launch(
                    cfg,
                    (
                        &b_dev,
                        n_max as i64,
                        &d_dev,
                        &c_dev,
                        n_cols,
                        n as i32,
                        dim as i32,
                        threshold,
                        modulus as i32,
                        ratio,
                        &mut o_rec,
                        &mut o_flags,
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        dev.synchronize().map_err(|e| e.to_string())?;
        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        stamp(&mut t_kernel);

        let len = n_cols_host;
        // NEGATIVE RESULT: collapsing these three arrays into ONE 16-byte
        // record per column (same size as DiameterEntryT, so it could be
        // reinterpreted with no conversion pass) measured SLOWER: 5.97 s ->
        // 6.27 s over two passes each. One 4.6 GB allocation plus one 4.6 GB
        // pinned buffer behaves worse than three smaller ones totalling
        // 4.0 GB. The obvious "one array, one memcpy" cleanup is not free at
        // multi-GiB scale.
        //
        // The page-lock of those 4.0 GB is ~1.3 s on its own, and it is
        // charged to this timer, so the buffers come from the process-level
        // cache: a grid-cells session runs 97 matrices in one process and
        // would otherwise pay it 97 times. See `cached_pinned_alloc`.
        // ONE buffer, ONE transfer, NO host-side copy.
        //
        // The kernel writes 16-byte records whose layout matches
        // `DiameterEntryT` bit for bit, so the page-locked buffer can be
        // read in place as `&[DiameterEntryT]`. The previous shape -- three
        // arrays, then a parallel pass that built a 4.6 GB
        // `Vec<DiameterEntryT>` -- cost 1.46 s and 4.6 GB of resident memory
        // for a copy nobody needed.
        let p_rec = cached_pinned_alloc(len * 16)?;
        let p_flags = cached_pinned_alloc(flag_words * 8)?;
        // SAFETY: the buffer holds `len` records and the device slice has
        // exactly `len` records.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(p_rec.as_ptr() as *mut i64, len * 2);
            memcpy_dtoh_sync(dst, *o_rec.device_ptr()).map_err(|e| e.to_string())?;
            let dst_f = std::slice::from_raw_parts_mut(p_flags.as_ptr() as *mut u64, flag_words);
            memcpy_dtoh_sync(dst_f, *o_flags.device_ptr()).map_err(|e| e.to_string())?;
        }
        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        stamp(&mut t_read);
        let idx = unsafe { p_rec.as_slice::<DiameterEntryT>(len) };
        #[cfg(any(feature = "apparent-stats", feature = "gpu-phase-stats"))]
        {
            stamp(&mut t_build);
            eprintln!(
                "GPUPHASE cols={len} dev={:.3} upload={:.3} alloc={:.3} kernel={:.3}                  pin_read={:.3} build={:.3} total={:.3} cols_htod={:.3}",
                t_dev.duration_since(t0).as_secs_f64(),
                t_upload.unwrap().duration_since(t0).as_secs_f64(),
                t_alloc.unwrap().duration_since(t_upload.unwrap()).as_secs_f64(),
                t_kernel.unwrap().duration_since(t_alloc.unwrap()).as_secs_f64(),
                t_read.unwrap().duration_since(t_kernel.unwrap()).as_secs_f64(),
                t_build.unwrap().duration_since(t_read.unwrap()).as_secs_f64(),
                t_build.unwrap().duration_since(t0).as_secs_f64(),
                t_cols_upload.duration_since(t0).as_secs_f64(),
            );
        }
        Ok(ApparentRecords {
            buf: Some(p_rec),
            flags: Some(p_flags),
            n: len,
        })
    }
}

#[cfg(not(feature = "gpu"))]
mod imp {
    use super::*;

    pub fn available() -> bool {
        false
    }

    pub fn last_error() -> String {
        "built without the `gpu` feature".to_string()
    }

    pub fn detect(
        _dense: &[f32],
        _n: IndexT,
        _n_max: usize,
        _binomial: &[IndexT],
        _cols: Vec<IndexT>,
        _dim: IndexT,
        _threshold: ValueT,
        _modulus: CoefficientT,
        _ratio: f32,
    ) -> Result<ApparentRecords, String> {
        Err("built without the `gpu` feature".to_string())
    }
}

/// The initialised device and module handle, for sibling GPU modules.
#[cfg(feature = "gpu")]
pub(crate) fn device_and_module() -> Result<
    &'static (
        std::sync::Arc<cudarc::driver::CudaDevice>,
        cudarc::driver::CudaFunction,
    ),
    String,
> {
    imp::device()
}

/// Whether a usable CUDA device is present. Cheap after the first call.
pub fn available() -> bool {
    imp::available()
}

/// Why the GPU path is unavailable, or empty when it is usable. Reported in the
/// STATS line because a silent CPU fallback looks exactly like "never tried".
pub fn last_error() -> String {
    imp::last_error()
}

/// Run apparent-pair detection for every column on the GPU.
///
/// `dense` must be the full `n * n` row-major matrix (the library keeps a
/// packed triangular layout whose row offsets are an indirection the GPU has no
/// use for, and the density is irrelevant at 5.76 MB for n=1200).
pub fn detect(
    dense: &[f32],
    n: IndexT,
    binomial: &BinomialCoeffTable,
    cols: &[DiameterIndexT],
    dim: IndexT,
    threshold: ValueT,
    modulus: CoefficientT,
    ratio: f32,
) -> Result<ApparentRecords, String> {
    // The kernel needs the simplex index of every column, and the host
    // holds it inside 16-byte records. Gathering the indices out of them
    // costs 2.4 GB of memory traffic (1.61 GB read + 807 MB written) -- a
    // SERIAL gather of 100.9M elements took 0.25 s, more than the PCIe
    // transfer it fed (0.077 s). A parallel gather reads the same bytes with
    // more memory-level parallelism, which is the point of the copy.
    #[cfg(feature = "parallel")]
    let idx: Vec<IndexT> = {
        use rayon::prelude::*;
        cols.par_iter().map(|c| c.get_index()).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let idx: Vec<IndexT> = cols.iter().map(|c| c.get_index()).collect();
    imp::detect(
        dense,
        n,
        binomial.n_max(),
        binomial.as_slice(),
        idx,
        dim,
        threshold,
        modulus,
        ratio,
    )
}

/// The 16-byte device-side record of the apparent pre-pass.
///
/// POD for cudarc, and laid out to match `DiameterEntryT` bit for bit:
/// the packed entry (index << 8 | coefficient) at offset 0, the apparent
/// partner's diameter at offset 8. The `apparent_entry_layout` unit test
/// pins the Rust side of that agreement, so a compiler that reorders the
/// struct fails the build instead of silently corrupting the data.
#[repr(C)]
#[derive(Clone, Copy)]
struct ApparentRec {
    entry: i64,
    diameter: f32,
}

/// A page-locked host buffer, for readbacks large enough that the driver's
/// pageable staging path dominates (measured: 2.0 GB/s pageable against
/// 24.2 GB/s pinned on this host).
///
/// Only worth it for REPEATED or large transfers -- a one-shot upload pays the
/// allocation (~0.73 s for 2.3 GB) and loses.
pub struct PinnedBuf {
    pub ptr: *mut std::os::raw::c_void,
    pub len: usize,
}

impl PinnedBuf {
    #[inline]
    pub fn as_ptr(&self) -> *mut std::os::raw::c_void {
        self.ptr
    }
}

#[cfg(feature = "gpu")]
impl Drop for PinnedBuf {
    fn drop(&mut self) {
        // SAFETY: ptr came from pinned_alloc and has not been freed.
        unsafe { crate::ripser::gpu::pinned_free(self.ptr) };
    }
}

/// Without the `gpu` feature nothing ever allocates a page-locked buffer, so
/// there is nothing to free. The type still exists because the apparent
/// population's container is compiled in both builds.
#[cfg(not(feature = "gpu"))]
impl Drop for PinnedBuf {
    fn drop(&mut self) {}
}

/// The apparent-pair population, exactly as the GPU produced it: a
/// page-locked array of `DiameterEntryT` records that the host reads IN
/// PLACE.
///
/// The detect kernel writes one 16-byte record per column, laid out to match
/// `DiameterEntryT` bit for bit (the layout is pinned by the
/// `apparent_entry_layout` unit test). So the reduction can use the
/// page-locked buffer directly as `&[DiameterEntryT]` and the host never
/// materialises a second copy of the population -- which cost 1.46 s and
/// 4.6 GB at ASA scale (the `d2_unaccounted` line of the phase breakdown).
///
/// The buffer is page-locked and owned here, so it goes back to the
/// process-level cache when this is dropped.
pub struct ApparentRecords {
    buf: Option<CachedPinned>,
    /// Two bits per column, packed into u64 words: bit 0 = "this column is an
    /// apparent pair", bit 1 = "this column also contributes a bar".
    flags: Option<CachedPinned>,
    n: usize,
}

impl ApparentRecords {
    /// An empty population (the CPU path's counterpart, and the state before
    /// the GPU pre-pass has run).
    pub fn empty() -> Self {
        Self {
            buf: None,
            flags: None,
            n: 0,
        }
    }
    /// Is this column an apparent pair (it never reduces)?
    #[inline]
    pub fn is_apparent(&self, i: usize) -> bool {
        self.flag(i, 0)
    }
    /// Does this column contribute a bar?
    #[inline]
    pub fn produces_bar(&self, i: usize) -> bool {
        self.flag(i, 1)
    }
    #[inline]
    fn flag(&self, i: usize, which: u32) -> bool {
        match &self.flags {
            None => false,
            Some(f) => {
                let words = unsafe { f.as_slice::<u64>((self.n + 31) / 32) };
                let w = i >> 5;
                let b = ((i & 31) * 2) as u32 + which;
                (words[w] >> b) & 1 == 1
            }
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    /// The population, read in place.
    #[inline]
    pub fn as_slice(&self) -> &[DiameterEntryT] {
        match &self.buf {
            None => &[],
            Some(b) => {
                debug_assert_eq!(std::mem::size_of::<DiameterEntryT>() * self.n, b.len());
                // SAFETY: the buffer holds `n` `DiameterEntryT` records and
                // was filled by an exact-length copy. The kernel writes the
                // same 16-byte layout (see `ApparentRec` in the .cu).
                unsafe { std::slice::from_raw_parts(b.as_ptr() as *const DiameterEntryT, self.n) }
            }
        }
    }
}

/// Allocate `len` bytes of page-locked host memory.
///
/// cudarc dlopens libcuda with a private handle, so the driver symbol has to
/// be resolved the same way rather than through `extern "C"` (which fails at
/// LINK time even though the symbol is in the process).
#[cfg(feature = "gpu")]
pub fn pinned_alloc(len: usize) -> Result<PinnedBuf, String> {
    type HostAllocFn = unsafe extern "C" fn(
        *mut *mut std::os::raw::c_void,
        usize,
        u32,
    ) -> cudarc::driver::sys::CUresult;
    // SAFETY: loading a system library that is already resident in the process
    // (cudarc has it open); the fn pointer is copied out before the handle goes.
    unsafe {
        let lib = libloading::Library::new("libcuda.so.1").map_err(|e| e.to_string())?;
        let sym: libloading::Symbol<HostAllocFn> =
            lib.get(b"cuMemHostAlloc\0").map_err(|e| e.to_string())?;
        let alloc: HostAllocFn = *sym;
        drop(sym);
        std::mem::forget(lib);
        let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
        let rc = alloc(&mut ptr, len, cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE);
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS || ptr.is_null() {
            return Err(format!("cuMemHostAlloc failed: {rc:?}"));
        }
        Ok(PinnedBuf { ptr, len })
    }
}

#[cfg(feature = "gpu")]
pub(crate) unsafe fn pinned_free(ptr: *mut std::os::raw::c_void) {
    type HostFreeFn =
        unsafe extern "C" fn(*mut std::os::raw::c_void) -> cudarc::driver::sys::CUresult;
    if let Ok(lib) = libloading::Library::new("libcuda.so.1") {
        if let Ok(sym) = lib.get::<HostFreeFn>(b"cuMemFreeHost\0") {
            (sym)(ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Process-level cache of page-locked host buffers.
//
// WHY. `cuMemHostAlloc` has to PAGE-LOCK the memory, and that is the
// expensive part, not the transfer. Measured on this machine:
//
//   pinned D2H of 4 GB        0.17 s   (24-26 GB/s)
//   cuMemHostAlloc of 4.3 GB  1.39 s   (0.32 s/GB)
//
// The GPU path allocates about 13.2 GB of page-locked memory per matrix --
// 4.0 GB for the apparent pre-pass readback and 9.2 GB for the assembly
// readback -- so it pays ~5 s of pure setup per matrix, and that cost is
// INVISIBLE in the phase breakdown because it is charged to whichever timer
// happens to enclose the allocation.
//
// A grid-cells session runs 97 matrices in ONE process, so paying that 97
// times is several minutes of setup that buys nothing. The cache keeps the
// buffers between calls and grows them on demand.
//
// The trade-off is memory: the cached buffers stay page-locked, so the
// process holds them until it exits. That is the right trade for a batch
// run and a poor one for an interactive session doing a single matrix, so
// `CANNS_RIPSER_PINNED_CACHE=0` turns it off.

thread_local! {
    static PINNED_CACHE: std::cell::RefCell<Vec<PinnedBuf>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// A page-locked buffer that returns itself to the cache when it goes out of
/// scope instead of being freed.
pub struct CachedPinned {
    buf: Option<PinnedBuf>,
    /// The number of bytes in use. The underlying allocation may be LARGER
    /// than this, because the cache hands a big buffer to a small request.
    len: usize,
}

impl CachedPinned {
    #[inline]
    pub fn as_ptr(&self) -> *mut std::os::raw::c_void {
        self.buf.as_ref().expect("buffer taken").ptr
    }
    /// The number of bytes allocated.
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.as_ref().map(|b| b.len).unwrap_or(0)
    }
    /// The usable bytes, as a typed slice.
    ///
    /// # Safety
    /// The caller must know the element type the buffer was filled with.
    pub unsafe fn as_slice<T>(&self, count: usize) -> &[T] {
        debug_assert!(std::mem::size_of::<T>() * count <= self.len);
        std::slice::from_raw_parts(self.as_ptr() as *const T, count)
    }
}

/// With the `gpu` feature the buffer goes back to the process-level cache;
/// without it there is no cache and the buffer is simply dropped (and the
/// `PinnedBuf` drop is a no-op there, because nothing ever allocated it).
#[cfg(feature = "gpu")]
impl Drop for CachedPinned {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            PINNED_CACHE.with(|c| c.borrow_mut().push(b));
        }
    }
}

#[cfg(not(feature = "gpu"))]
impl Drop for CachedPinned {
    fn drop(&mut self) {
        self.buf.take();
    }
}

/// Allocate `len` bytes of page-locked memory, reusing a cached buffer when
/// one is big enough.
///
/// The reuse policy is BEST FIT: the smallest cached buffer that can hold the
/// request, so a 4 GB buffer is not spent on a 16-byte request.
pub fn cached_pinned_alloc(len: usize) -> Result<CachedPinned, String> {
    let use_cache = !std::env::var("CANNS_RIPSER_PINNED_CACHE")
        .map(|v| matches!(v.trim(), "0" | "false" | "False" | "no" | "off"))
        .unwrap_or(false);
    let reused = if use_cache {
        PINNED_CACHE.with(|c| {
            let mut v = c.borrow_mut();
            let mut best: Option<usize> = None;
            for (i, b) in v.iter().enumerate() {
                if b.len >= len && best.map(|j| v[j].len > b.len).unwrap_or(true) {
                    best = Some(i);
                }
            }
            best.map(|i| v.remove(i))
        })
    } else {
        None
    };
    let buf = match reused {
        Some(b) => b,
        None => {
            #[cfg(feature = "gpu")]
            {
                pinned_alloc(len)?
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err("built without the `gpu` feature".to_string());
            }
        }
    };
    Ok(CachedPinned {
        buf: Some(buf),
        len,
    })
}

/// Materialise the dense matrix the kernel indexes. O(n^2) `get` calls, which is
/// ~0.7M at n=1200 and therefore noise next to the kernel itself.
pub fn densify<M: DistanceMatrix>(dist: &M, n: IndexT) -> Vec<f32> {
    let nn = n as usize;
    let mut out = vec![0f32; nn * nn];
    for i in 0..nn {
        for j in (i + 1)..nn {
            let d = dist.get(i, j);
            out[i * nn + j] = d;
            out[j * nn + i] = d;
        }
    }
    out
}

impl From<&ApparentResult> for DiameterEntryT {
    fn from(r: &ApparentResult) -> Self {
        if r.index == -1 {
            DiameterEntryT::new(-1.0, -1, 0)
        } else {
            DiameterEntryT::new(r.death, r.index, r.coeff)
        }
    }
}

/// Sentinel for a column that produced nothing.
pub fn sentinel() -> DiameterEntryT {
    DiameterEntryT::from(&ApparentResult::sentinel())
}
