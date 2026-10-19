//! A global allocator that asks the kernel for transparent huge pages on
//! large allocations.
//!
//! Why this exists
//! ---------------
//! The pivot map in `compute_pairs` is the single largest allocation in the
//! library. Measured on the real ASA matrix at dim=2 (standalone
//! micro-benchmark, 286,562,199 inserts, identical arithmetic in both arms):
//!
//! ```text
//! big  (~13.1 GB): 31.96 s   111.5 ns/insert   3,277,686 minor page faults
//! small (~6 MB)  :  2.47 s     8.6 ns/insert
//! ```
//!
//! So 103 of the 112 ns per insert is memory-system cost, not hashing. Two
//! thirds of that bill is addressable without touching the algorithm:
//!
//! * 3.28M first-touch faults are 13.1 GB at 4 KB per page. At 2 MB pages the
//!   same footprint needs ~6,700 faults.
//! * A 13.1 GB working set needs ~419K 4 KB pages to cover, which no TLB holds,
//!   so almost every probe misses. 2 MB pages cut that by 512x.
//!
//! Both are decided by the kernel only for regions explicitly marked, and this
//! server runs `transparent_hugepage/enabled = madvise` -- the mode where
//! `MADV_HUGEPAGE` per VMA is the intended mechanism.
//!
//! Scope of the change: allocations at or above [`HUGE_THRESHOLD`] only, so the
//! many small Vec pushes in the reduction loop keep their ordinary 4 KB pages.
//! `defrag` is `madvise` on this host, so marked regions are compacted on
//! demand; with ~99 GB free that is cheap, but it is the reason this is opt-in
//! rather than unconditional.

use std::alloc::{GlobalAlloc, Layout, System};

/// Linux `MADV_HUGEPAGE`. Spelled out rather than pulled from `libc` to keep
/// the dependency tree (and therefore the wheel build) untouched.
#[cfg(target_os = "linux")]
const MADV_HUGEPAGE: i32 = 14;

#[cfg(target_os = "linux")]
extern "C" {
    fn madvise(addr: *mut core::ffi::c_void, length: usize, advice: i32) -> i32;
}

/// One huge page. Below this the syscall's cost outweighs any benefit: these
/// allocations are served from already-mapped pages in cache.
pub const HUGE_THRESHOLD: usize = 2 * 1024 * 1024;

pub struct HugePageAlloc<A>(pub A);

#[cfg(target_os = "linux")]
#[inline]
unsafe fn hint(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len < HUGE_THRESHOLD {
        return;
    }
    // madvise returns EINVAL for ranges that are not page aligned. glibc serves
    // large blocks with mmap, so the start is page aligned in practice, but
    // rounding up costs nothing and makes the hint unconditional.
    const PAGE: usize = 4096;
    let start = ptr as usize;
    let aligned = (start + PAGE - 1) & !(PAGE - 1);
    let end = (start + len) & !(PAGE - 1);
    if end <= aligned {
        return;
    }
    // Safety: the range is inside a live allocation owned by the caller.
    unsafe {
        madvise(
            aligned as *mut core::ffi::c_void,
            end - aligned,
            MADV_HUGEPAGE,
        );
    }
}

#[cfg(not(target_os = "linux"))]
#[inline]
unsafe fn hint(_ptr: *mut u8, _len: usize) {}

unsafe impl<A: GlobalAlloc> GlobalAlloc for HugePageAlloc<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.0.alloc(layout) };
        unsafe { hint(p, layout.size()) };
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.0.alloc_zeroed(layout) };
        unsafe { hint(p, layout.size()) };
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.0.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { self.0.realloc(ptr, layout, new_size) };
        // A realloc that grows into the huge-page range needs the hint again;
        // the old mapping is gone either way, so re-marking is the only chance.
        unsafe { hint(p, new_size) };
        p
    }
}

pub type SystemHugePages = HugePageAlloc<System>;
