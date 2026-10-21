//! Compiles `kernel/ripser_kernels.cu` with nvcc into a cubin (preferred) and a
//! ptx (fallback), both embedded into the crate for runtime loading.
//!
//! Pattern follows `flashbrain2/src/flashbrain/build.rs`, including the two
//! failure modes that cost real debugging time there:
//!
//! 1. **cubin and ptx fail in different places.** A cubin is machine code for
//!    one exact architecture: `cuModuleLoadData` takes it directly with no JIT,
//!    so the driver's CUDA version is irrelevant. A ptx is JIT-ed by the driver
//!    at load time and is rejected outright when a newer toolkit produced it
//!    (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) -- the flashbrain2 A6000 case:
//!    driver 550 (CUDA 12.4) against a 12.5 toolkit.
//!
//! 2. **A stale cubin is silent and the log lies.** Cargo does not track
//!    environment variables that `build.rs` reads unless they are declared. So
//!    changing `NVCC_ARCH` without `rerun-if-env-changed` leaves the previous
//!    architecture's cubin in `OUT_DIR` while the log cheerfully prints the new
//!    one; the mismatch only surfaces at runtime as "invalid-image". Hence the
//!    explicit `rerun-if-env-changed` below.
//!
//! Neither cubin nor ptx is required to BUILD. If nvcc is absent the embedded
//! images are empty and the runtime uses the NVRTC path in `gpu.rs` /
//! `gpu_assembly.rs`, which needs no nvcc. That keeps the `gpu` feature from
//! becoming a hard build requirement for anyone.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Watch the DIRECTORY, not one file: kernel/ is a single .cu today, but
    // cargo treats a directory as "any file under it changed", which is what
    // we want if a .cuh ever joins it.
    println!("cargo:rerun-if-changed=kernel");

    // ⚠ Both knobs participate in the rebuild decision. Without these lines an
    // arch change (or a skip flag) leaves the previous image in OUT_DIR and the
    // log prints the architecture that was *intended*, not the one on disk.
    println!("cargo:rerun-if-env-changed=NVCC_ARCH");
    println!("cargo:rerun-if-env-changed=CANNS_RIPSER_SKIP_CUDA");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    if std::env::var("CANNS_RIPSER_SKIP_CUDA").is_ok() {
        println!("cargo:warning=CANNS_RIPSER_SKIP_CUDA set -> embedding empty images");
        emit(&out_dir);
        return;
    }

    // macOS has no CUDA (Apple GPUs are Metal) and CI runners frequently have
    // no toolkit. Neither is an error: the runtime picks the NVRTC path or the
    // CPU path, exactly as it did before this file existed.
    if !cfg!(target_os = "linux") && !cfg!(target_os = "windows") {
        println!(
            "cargo:warning=canns-lib: CUDA kernels skipped on this target; \
                  the GPU path will fall back to NVRTC or CPU"
        );
        emit(&out_dir);
        return;
    }

    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| {
        for cand in ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"] {
            if Path::new(cand).exists() {
                return cand.to_string();
            }
        }
        "nvcc".to_string()
    });

    // Default to the oldest architecture that runs everywhere we support, so a
    // plain `cargo build` produces a loadable cubin instead of one that only
    // works on the build machine.
    let arch = std::env::var("NVCC_ARCH").unwrap_or_else(|_| "sm_70".to_string());

    println!("cargo:warning=canns-lib: nvcc kernel/ripser_kernels.cu -> cubin ({arch})");

    let cubin = Command::new(&nvcc)
        .args(["-O3", "-Xptxas", "-v", &format!("-arch={arch}")])
        .args(["-cubin", "-o"])
        .arg(out_dir.join("ripser_kernels.cubin"))
        .arg("kernel/ripser_kernels.cu")
        .output();

    match cubin {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            println!(
                "cargo:warning=canns-lib: nvcc failed to build the cubin (exit {:?}, arch={arch}). \
                 Falling back to the NVRTC path, which needs no nvcc but is JIT-ed by the \
                 driver and can therefore fail with CUDA_ERROR_UNSUPPORTED_PTX_VERSION.\n{}",
                o.status.code(),
                String::from_utf8_lossy(&o.stderr)
            );
            emit(&out_dir);
            return;
        }
        Err(e) => {
            println!(
                "cargo:warning=canns-lib: cannot run nvcc ({nvcc}): {e}. \
                 The GPU path falls back to NVRTC (no nvcc needed) or to the CPU. \
                 If you want the cubin path, put nvcc on PATH or set NVCC."
            );
            emit(&out_dir);
            return;
        }
    }

    let ptx = Command::new(&nvcc)
        .args(["--ptx", "-O3", &format!("-arch={arch}"), "-o"])
        .arg(out_dir.join("ripser_kernels.ptx"))
        .arg("kernel/ripser_kernels.cu")
        .output();

    match ptx {
        Ok(o) if o.status.success() => {}
        Ok(_) | Err(_) => {
            // ptx is only the fallback for "cubin exists but its arch does not
            // match this machine". Losing it is not fatal, but say so.
            println!(
                "cargo:warning=canns-lib: ptx fallback failed to build; \
                      the GPU path will rely on the cubin alone"
            );
        }
    }

    emit(&out_dir);
}

/// Write the files the crate reads at compile time.
///
/// * the two kernel IMAGES (`include_bytes!`), empty when the build had no
///   nvcc -- that emptiness is how the runtime learns to take the NVRTC path;
/// * the kernel SOURCE (`include_str!`), always the real `kernel/ripser_kernels.cu`.
///
/// The source copy is what makes the NVRTC fallback compile the SAME kernels
/// as the nvcc path. It used to be two hand-maintained Rust string literals
/// (one in gpu.rs, one in gpu_assembly.rs, the latter 417 lines) that were
/// verified to be byte-identical to the .cu at this revision -- but nothing
/// kept them that way. One `include_str!` removes the duplicate entirely and
/// makes "the NVRTC path and the PTX path run the same code" a property of the
/// build rather than a convention.
fn emit(out_dir: &Path) {
    for name in ["ripser_kernels.cubin", "ripser_kernels.ptx"] {
        let path = out_dir.join(name);
        if !path.exists() {
            std::fs::write(&path, b"").expect("write empty kernel image");
        }
    }
    let src = Path::new("kernel/ripser_kernels.cu");
    let dst = out_dir.join("ripser_kernels.cu");
    if src.exists() {
        std::fs::copy(src, &dst).expect("copy kernel source into OUT_DIR");
    } else {
        std::fs::write(&dst, b"").expect("write empty kernel source");
    }
}
