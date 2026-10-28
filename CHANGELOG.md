# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance
- **The cofacet enumerator stops reading distances for a cofacet the threshold
  already rejects -- and only does so when that pays.** The enumerator used to
  read all `dim + 1` distances of a candidate cofacet before the caller could
  reject it on the threshold. On `o3_4096` (4,096 points, threshold 1.4) only
  1.3% of the candidates are under the threshold, so 98.7% of the distance reads
  were spent on cofacets that were thrown away -- and each of those reads is a
  scattered miss into a 33.5 MB matrix. The threshold is now passed into the
  enumerator and the read loop stops as soon as the running maximum is above
  it; the true diameter is at least the running maximum, so such a cofacet is
  rejected either way, and all seven call sites filter on the threshold before
  they use the diameter.

  **The switch is gated on the rejection rate, because the sign of its effect
  is a property of the data.** Measured on the same binary, the same host and
  the same 5.76 MB matrix, with the arm order reversed:

  | threshold | early exit off | early exit on | |
  |---|---|---|---|
  | T (the maximum finite distance) | 12.63 s | 14.67 s | 1.16x, a loss |
  | T/2 | 7.92 s | 10.48 s | 1.32x, a loss |
  | T/4 | 1.15 s | 1.04 s | 0.90x, a win |
  | T/10 | 0.65 s | 0.27 s | 0.41x, a win |

  and 1.9x faster on `o3_4096`, where 98.7% of the candidates are rejected and
  every read it skips is a DRAM miss. The early exit saves the distance reads
  of the candidates the threshold rejects and costs one comparison per read, so
  it pays when most candidates are rejected and it LOSES when they are not: on
  the scale scan and on the ASA matrix, where the threshold is the matrix's
  maximum distance and only the missing pairs are rejected, it was a 16-21% loss.
  The gate is the rejection rate, and the algorithm layer knows it exactly --
  the filtration holds the edges the threshold KEEPS, and every other pair is
  one it rejects. The fraction is computed once per call, before the
  reduction, and the enumerator asks for it once per walk rather than per read.

  `CANNS_RIPSER_COFACET_EARLY_EXIT=0` restores the old behaviour exactly and
  `=1` forces the early exit on whatever the data says, so the switch itself
  can still be measured in isolation.

  The barcode is unchanged: the two arms of the switch produce the same bars,
  and the old and the new build produce byte-identical barcodes on all 99
  inputs of the cross-build check (`o3_4096`, `asa_1200` and all 97 corpus
  matrices). The reduction rounds per dimension are identical between the arms
  as well -- the arms do the same work, and the change is in the cost of doing
  it.

  **Where this applies, stated exactly.** The early exit is in the HOST-side
  cofacet enumerator, which is the reduction's path in every build, GPU
  included: the reduction runs on the host whatever the build. It is **not**
  in the GPU's column-assembly kernels, which compute the cofacet diameter
  themselves (`kernel/ripser_kernels.cu` reads the `dim + 1` distances and
  then tests `cd <= threshold`), so the assembly phase keeps the old
  behaviour. The change makes the reduction cheaper; it does not make the
  assembly cheaper.
- **One comparison per distance read, not two, and the loop's state is hoisted
  out of it.** The rejection test used to be two comparisons: `d > threshold`
  and then the infinite case. With an infinite threshold the first could never
  fire, so it was pure overhead on every read -- and the reads it was paying
  for were the cheapest kind there are. The enumerator now computes, once per
  matrix, the value a distance has to reach for the cofacet to be rejected --
  the next float above a finite threshold, or infinity for an infinite one --
  and one comparison covers both rejections, and the `-inf` case with them.
  The test is on the distance just read rather than on the running maximum,
  which is the same test (the running maximum can only reach `reject_at`
  through a READ, because the simplex's own diameter is <= the threshold) and
  one link shorter on the critical path. The loop's fields are copied into
  locals first, because the compiler cannot prove that a call through the
  matrix does not write through `self` and so reloads them on every iteration.
  Measured on the 1,200-vertex scale-scan instance: 15.16 s against 15.41 s
  before the change, and 12.61 s with the early exit off.
- **A GPU build no longer creates a CUDA context to answer a question about
  the GPU.** `gpu::last_error()` called `device()`, and `device()` creates the
  context -- and the reduction called `last_error()` on every call, to fill in
  one statistics field. Every call of a GPU build therefore paid ~0.245 s for
  a context it might never use, including the calls whose work was far too
  small to be worth the device. The error is now recorded when the device is
  initialised, so asking for it is free: the first call on a small instance
  goes from 0.248 s to 0.006 s, the same as the CPU build.
- **The GPU apparent pre-pass is only taken when it repays the device.** The
  context costs ~0.245 s per process and the pre-pass saves ~60 ns per
  column, so the device only pays for itself at about four million columns --
  and the grid-cell corpus runs at dimension 1 on 1200-vertex matrices, 500K
  columns, where the pre-pass saves 40 ms and the device costs 245 ms. That is
  why the corpus was 25% slower on the GPU build than on the CPU build
  (1.366 s against 1.095 s, against ripser++'s 2.015 s): the device was paid
  for per cell, and the corpus measures one cell per process. The count is
  cumulative over the process's calls, because the context is paid once -- a
  process that makes many calls on small instances earns the device, and one
  that makes a single call on a small instance does not pay for it. The corpus
  cells now measure the same on both builds (1.025 s against 1.019 s on the
  first matrix), with identical barcodes.
- **The GPU apparent pre-pass no longer copies the column indices twice.** The
  kernel needs the simplex index of every column, and the host holds it inside
  16-byte records, so the indices were gathered out into a `Vec<i64>` and then
  copied AGAIN by `cols.to_vec()` on the way into the driver (which keeps that
  copy as the device slice's host buffer). Measured at dim=2 on the 1200x1200
  ASA matrix: the serial gather cost 0.25 s, the second copy 0.23 s, and the
  PCIe transfer they fed 0.077 s -- two thirds of the cost of getting 807 MB
  to the GPU was host-side copying. The gather is now parallel and the second
  copy is gone: the vector is moved into the device slice, so the driver reads
  it in place. The pre-pass goes from 0.667 s to 0.236 s, the barcode is
  byte-identical, and the head-to-head against ripser++ goes from 1.050x to
  0.918x.
- **The reduction's working column store keeps one bound per NON-EMPTY
  column.** The store used to hold one `u32` per ENUMERATED column -- the end
  offset of that column's entries -- and `append_column` ran once per
  enumerated column: 286,562,199 times at dim=2 on the 1200x1200 ASA matrix.
  That is 1.15 GB of bounds at dim=2 (403 MB at the T threshold), and every one
  of those entries was equal to its predecessor's, because the overwhelming
  majority of the columns are apparent pairs and store nothing at all. The
  store now keeps one `(column, end offset)` pair per non-empty column --
  298,670 of them at dim=2, 2.4 MB -- and finds a column's range with a binary
  search. The semantics are unchanged (`subrange(c)` returns exactly the
  entries pushed while column `c` was being processed, and an empty column
  returns an empty range), and the barcode is byte-identical on every arm of
  the head-to-head. Measured on the ASA matrix, three interleaved repetitions
  per arm: 1.1% to 2.0% faster on all four arms, and the peak resident set
  drops by 382 MB (T threshold) to 1.06 GB (cap threshold).
- **The reduction's working column heap is now a hand-rolled `CoboundaryHeap`.**
  `std::collections::BinaryHeap::pop` uses `sift_down_to_bottom`, which always
  walks the hole to the bottom of the tree. A single column's coboundary heap
  grows to 9-11 million entries (148-185 MB) before it is drained, so the
  bottom of the tree is a cold cache line on every pop; ripser++ uses
  `std::priority_queue`, whose `pop` is the classic top-down sift-down with an
  early exit. The new heap implements that variant and is the default; the
  `binary-heap` cargo feature restores `BinaryHeap`. The unit test
  `pop_order_matches_binary_heap` pins the pop order against `BinaryHeap`, and
  `pops_in_reduction_order` pins the reduction order.
- **The modulus cancellation in the reduction no longer divides.** The two
  coefficients are always smaller than the modulus, so a single conditional
  subtraction replaces the integer division. The coefficients are also summed
  in `i32` rather than `i16`; the old `i16` addition overflowed for a modulus
  above 16383.
- **`import canns_lib` no longer imports scikit-learn eagerly.**
  `sklearn.metrics.pairwise.pairwise_distances` is only needed when the caller
  passes points instead of a distance matrix, and importing it at module scope
  cost 0.96 s on every import.
- **Maturin builds now enable the `hugepage` cargo feature** on Linux. It marks
  allocations of 2 MB or more with `MADV_HUGEPAGE`, which is worth 1.24x end to
  end on the real ASA workload (119.4 s -> 96.0 s, three interleaved passes,
  bitwise-identical barcode on all 224 regression cells) because the pivot map
  is a single ~13 GB allocation whose insert cost is dominated by page faults
  and TLB misses rather than hashing. It is a no-op on non-Linux. Build with
  `--no-default-features` (or drop the feature) to opt out.

### Changed
- **The benchmark harness is a tree, and it has a written rule.** A file in
  `benchmarks/ripser/` has to be a runner, a library, a driver, a tool or a
  patch; a probe whose question has been answered is none of those, and it
  moves to the experiment record. The tree is `harness/` (the runners, with
  the libraries they import beside them), `drivers/` (the shell helpers),
  `tools/` (the things that work on tables rather than produce them),
  `patches/` (applied to a COPY of ripser++) and `archive/` (the probes that
  still guard something). A runner and the library it imports stay in the same
  directory, so the imports work wherever the tree is checked out.
- **The quiet-window gate has one implementation, and it is in the shared
  core.** The benchmark host is shared, a neighbour's job moves the numbers by
  up to 40%, and it can flip a comparison's sign -- so a table measured on a
  busy host is not a table with a caveat, it is not a table. The gate is the
  number of cores used by processes that are NOT ours, read from `/proc` while
  the row was being measured, and it excludes our own work by construction, so
  it needs no allowance for how busy the experiment is. It lives in
  `harness/bench_core.py` (`foreign_summary` and `batch_is_clean`), and the
  drivers, the fetch script and the corpus chain all call it. A table with no
  foreign-load reading FAILS, with the reason spelled out.
- **The gate tells a crashed row from a timed-out row.** A run that hit the
  time limit is a legal result -- the runners record it as `TIMEOUT` -- and a
  table that says "this did not finish inside 600 s" says the truth. A row
  whose child DIED is not a measurement: it looks like a fast run and it is
  not. The two are counted separately and reported separately, because a gate
  that calls both "contaminated" sends the reader to the wrong place.
- **The runners that produce tables sample the host's load while the child
  runs.** `harness/bench_core.py`'s `run_measured` wraps a runner's existing
  call -- the same command, the same `/usr/bin/time` wrapper, the same
  parsing -- and adds the reading, so the table gains the column that proves
  it was measured in a quiet window without any of its numbers changing
  meaning.
- **One measurement at a time.** The quiet-window gate counts the cores used
  by processes that are NOT ours, so it cannot see two of OUR OWN measurements
  colliding -- and that collision moves the numbers just as much as a
  neighbour's job does. It happened: a measurement driver's parent was stopped
  while its child kept running, and the child and an unrelated measurement ran
  together for half an hour. Every measured run now takes an exclusive lock on
  the box; a second measurement waits for the first rather than running beside
  it.
- **The corpus table's arm list has one definition.** It lived in two places
  -- the corpus' own driver, which is in no repository because the corpus
  pipeline's licence is still open, and the code here -- and the two copies
  were coupled by hand. The list is now `harness/corpus_arms.py`, and the
  driver and the analysis read it. It also says what the arms are FOR: the
  threshold is a table-level constant, every arm computes the same filtration,
  and the arm with ripser++ at the cap threshold is deliberately not in it,
  because it admits every pair in the file as a real edge and so computes a
  different filtration.
- **The sync to the benchmark host now deletes what no longer exists here.**
  The host's tree is not a git checkout, so a file that was removed from this
  repository stayed there and kept running, and the set of files to remove is
  now computed as "what is on the host and not here", never as a hard-coded
  list of names. The sync also skips the AppleDouble files that a macOS
  checkout leaves next to the real ones, and it copies the RUST sources
  separately, with an explicit file list, because a benchmark script that did
  not land fails loudly while an engine file that did not land compiles into a
  binary that runs and produces numbers, which fails silently.

### Fixed
- **Five benchmark runners could not run at all.** The commit that made the
  runners that spawn their own children take the measurement lock added two
  lines to five *embedded child scripts*: `sys.path.insert(0,
  os.path.dirname(os.path.abspath(__file__)))` and `import bench_core`. The
  children use `os` and none of them import it, and none of them need
  `bench_core` either, so every one of them died with a `NameError` on its
  first line of work -- and the parent had the same mistake, so the parent
  died on `bench_core.measurement_lock`. The scale scan therefore produced a
  table of fifteen rows in which our nine rows were `CRASH`, and the
  quiet-window gate reported the batch as "contaminated", which points at the
  host instead of at the bug. The two lines are gone from the children, the
  parent imports the module it uses, and the static check
  (`benchmarks/ripser/tools/smoke_children.py`) now refuses a child script
  that uses a name it never binds.
- **The corpus chain no longer dies with the table it just produced.** The
  corpus driver is followed by a check, and the chain runs under `set -e`, so
  a check that returned non-zero took the whole chain down with it -- and the
  two tables after it were never produced. A step that fails is now recorded
  in the log and the chain goes on to the next one.
- **The ripser distance-matrix path no longer takes the sparse route on a
  dense matrix.** `rips_dm_with_callback_and_interval` switched to the sparse
  distance matrix whenever `threshold < max_finite`. Any pipeline that
  replaces missing (`+inf`) distances with a finite cap -- the ASA pipeline
  caps them at `2*max_finite + 1` -- therefore raised `max_finite` to the cap,
  and every threshold below the cap satisfied the condition, so an ordinary
  dense matrix was walked down the sparse path. In this implementation the
  sparse path is the slower one: the walk counters and the barcode are
  identical on both routes, but the sparse arm took 4.3x as long, because the
  dense enumerator rejects a capped pair with a single comparison while the
  sparse enumerator walks adjacency lists. The decision now requires the
  matrix to actually contain missing pairs. Measured on the 1200x1200 ASA
  matrix: the capped arm went from 49.419 s to 11.979 s, the other three arms
  were unchanged.

### Removed
- **The benchmark tree's top level is gone.** The two scripts that sat in
  `benchmarks/ripser/`'s top level moved into it: `comprehensive_benchmark.py`
  is a runner and is now `harness/`, `analysis_results.py` is a tool and is now
  `tools/`. The shuffle protocol `ASA_SHUFFLE.md`, the two one-off probes
  `phase_baseline.py` and `progress_bar_impact.py`, and the sample PNG in
  `analysis/` left with them.

### Added
- **A reproducible head-to-head harness against ripser++** in
  `benchmarks/ripser/`: the runner, the load characterisation over the 97
  grid-cell matrices, the standard suite on ripser++'s own datasets, the scale
  scan and the unified ablation. Every table is produced by a script, and
  every row records the number of cores used by processes that are not ours
  while that row was being measured, so a table that cannot be shown to have
  been measured on a quiet host is not a table. The tables are not committed:
  the grid-cell corpus behind two of them has an unsettled licence, so only the
  scripts are published until it is.
- **`gpu` cargo feature** -- an opt-in GPU apparent-pair pre-pass, off by
  default and **not in the release build**: `[tool.maturin]` enables only
  `parallel`, `hugepage` and `extension-module`, so the published wheels
  contain no CUDA code at all and the pre-pass runs on the CPU. The GPU
  pre-pass is a build, not a download:
  `maturin develop --release --features parallel,hugepage,extension-module,gpu`
  (keep `extension-module` -- it is what makes PyO3 produce a cdylib for
  Python). A run with no usable device falls back to the CPU pre-pass, and the
  barcode does not change. Measured on an A100-SXM4-80GB with the same
  2.9e8 columns as the ASA matrix: the CPU pre-pass takes 12.0-15.5 s against
  0.093 s for the GPU kernel plus 0.095 s for the pinned readback. NOTE that
  building this feature currently needs a CUDA toolkit: our own `build.rs`
  compiles the kernel with `nvcc` only when one is present, but the `cudarc`
  dependency's build script runs `nvcc --version` to detect the CUDA version
  and fails without it.
- **`benchmarks/ripser/tools/smoke_children.py`** -- a static check of the
  embedded child scripts: for every runner, it reports the names a child uses
  but never binds. It needs no numpy, no site directory and no device, so it
  runs in a second, and it runs *before* a measurement rather than after it.
  It exists because five runners were broken in exactly that way and the only
  symptom was a table full of crashes.

## [0.11.0] - 2026-09-23

### Breaking
- The private `canns_lib._ripser_core.shuffle_null_model` neuron-distance algorithm has been removed. The retained private name raises a migration `ValueError`. The public `canns_lib.ripser.shuffle_null_model` now exposes explicit metric and persistence parameters for a feature-space null: shift feature columns, compute distances between rows, then run Ripser. It does not accept `pipeline` or `pipeline_kwargs` and does not perform full ASA preprocessing. Precomputed distance matrices are rejected because independent column shifts do not preserve their symmetry. See [the migration guide](docs/shuffle.md#migration-from-the-private-native-shuffle).
- Full ASA shuffle remains in CANNs with typed `TDAConfig` and the same complete analysis for real and shuffled data. Downstream CANNs must include the coordinated update in [companion PR #103](https://github.com/Routhleck/canns/pull/103). As of 2026-09-20, no released CANNs version includes it; there is no released minimum version to pin yet. Do not upgrade only canns-lib for an existing CANNs deployment that uses the private native shuffle.

### Added
- Public `generate_offsets(shape, num_shuffles, *, seed=None, shifts=None)` for reproducible, independently shifted feature columns. Returns a read-only int64 offset matrix; supports exact replay without changing global NumPy random state.

## [0.10.2] - 2026-09-17

### Fixed
- **Segfault on macOS Darwin 26 (kernel 25.5+)** caused by the default `mimalloc 0.1.43` global allocator crashing inside `mi_bbitmap_try_find_and_clear_generic` during the first pyo3 thread allocation. lldb backtrace: `EXC_BAD_ACCESS (code=2, address=0x16edfffd0)` on any `ripser()` call — even a 3-point smoke test exits with signal 11. CI runners (ubuntu/windows/macos-13/14) never hit this because they run on older Darwin kernels; macOS users on 26.x see silent breakage. The default build now uses the system allocator, which works on every platform; opt back into mimalloc via `maturin develop --features mimalloc` (or `cargo build --features mimalloc`) if you want the (small) allocation-throughput boost on Linux/Windows.
- **`cargo build --lib` and `cargo test --lib` fail with `no method named 'into_pyarray' found for ndarray 0.16`** when the top-level dep is pinned to a version different from numpy 0.28's transitive ndarray. Fresh `cargo build` (without a pre-existing `Cargo.lock`) resolves TWO ndarray versions — 0.16.x from our crate and 0.17.x from numpy 0.28. The two `ArrayBase` types are distinct, so numpy's `impl<A, D> IntoPyArray for ArrayBase<OwnedRepr<A>, D>` is invisible to our `array.into_pyarray(py)` call. The fix is to pin top-level `ndarray` to the same major version as numpy 0.28's transitive requirement (currently `"0.17"`). `pip install -e .` via maturin masks the bug because maturin keeps an internal Cargo.lock that resolves consistently.

### Added
- **`mimalloc` cargo feature** — opt-in, off by default. `maturin develop --features mimalloc` (or `cargo build --features mimalloc`) enables the mimalloc global allocator. See the Cargo.toml note on the `mimalloc` dep for platform guidance.

## [0.10.1] - 2026-08-08

### Fixed
- **`cibuildwheel` upgraded from 2.21.0 to 4.2.0** so the release workflow actually builds `cp314-*` wheels. v2.21 (Nov 2024) was published four months *before* Python 3.14 GA, so its `build` glob parser didn't recognize `cp314-*` as a valid identifier and silently skipped all of them — that's why the v0.10.0 PyPI release had 22 files but zero cp314 wheels despite `CIBW_BUILD: cp311-* cp312-* cp313-* cp314-*` in `.github/workflows/release.yml`. cibuildwheel 4.0.0 (Jun 2026) is the first release where cp314 is built by default.
- No release.yml or `CIBW_ENVIRONMENT_MACOS` rewrite needed: v4.2.0 still accepts `CIBW_ENVIRONMENT_MACOS` (it's in the [options docs](https://cibuildwheel.pypa.io/en/stable/options/#environment) as `_MACOS | _WINDOWS | _LINUX | _ANDROID | _IOS | _PYODIDE`). The only behavior change is `delvewheel` becoming the default Windows repair command in 4.0+ — that just bundles any pyo3 extension `.pyd` DLLs in the wheel, which is the right thing for canns-lib anyway.
- `Cargo.lock` is .gitignored (project is a library), so the wheel-build matrix is determined by the `Cargo.toml` `pyo3 = "0.28.2"` / `numpy = "0.28.0"` / `ndarray = "0.17"` triple plus the transitive constraint union — cibuildwheel 4.2.0 on fresh CI resolves the same `pyo3 0.28.3 / numpy 0.28.0 / ndarray 0.17.2` graph that v0.10.0 was built with, so no source changes are needed.

## [0.10.0] - 2026-08-08

### Added
- **Python 3.14 support.** Wheels for `cp314-*` are now published on PyPI; CI matrix adds a `3.14` job on Ubuntu, macOS, and Windows. `pyproject.toml` classifiers gain `Programming Language :: Python :: 3.14`. Uses `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` (already enabled) so the single abi3 wheel covers 3.11–3.14.

### Changed
- **PyO3 upgraded from 0.24.2 to 0.28.x** (resolved to 0.28.3 at lock time). PyO3 0.25–0.27 added Python 3.14 support; 0.27 was the first release tested against 3.14.0 final; 0.28 fixes an abi3 subclassing soundness bug. `pyo3-build-config` upgraded 0.20 → 0.28 in lockstep.
- **`rust-numpy` upgraded from 0.24.0 to 0.28.0** (resolved to 0.28.0 at lock time; 0.29 would require PyO3 0.29). Rust MSRV raised to 1.83 (matches PyO3 0.28 and `rust-numpy` 0.28). Local dev environment should be fine — current stable toolchains are 1.85+.
- **`ndarray` bumped from 0.16 to 0.17** (resolved to 0.17.2). numpy 0.28 keeps its `ndarray = ">= 0.15, <= 0.17"` upper bound, and 0.17.x is the only common ground between canns-lib and the rest of the (transitively-pinned) numpy 0.28 / pyo3 0.28 dep graph. Without this bump, fresh `cargo` resolves (i.e. CI, which doesn't read the local `Cargo.lock` because the project is a library) split ndarray into 0.16.1 (canns-lib) and 0.17.2 (numpy 0.28), which broke the `IntoPyArray` trait resolution on `ndarray::ArrayBase`. Zero code changes — only `src/spatial/{agent,environment}.rs` use `ndarray::Array2`, which is unchanged in 0.17.

### Breaking (Rust API)
- `pyo3::Python::with_gil(|py| { ... })` is renamed to `pyo3::Python::attach(|py| { ... })` in 5 call sites under `src/ripser/core/{algorithm,reduction}.rs` (PyO3 0.26 rename, kept working with deprecation until 0.28). Behavior is identical — `attach` is the GIL-aware counterpart of the same closure.
- `pyo3::PyObject` is replaced with `pyo3::Py<pyo3::PyAny>` in 13 call sites across `src/ripser/core/{algorithm,reduction}.rs` and `src/ripser/mod.rs` (PyO3 0.26 deprecated the type alias, 0.28 removed it). All occurrences are in `progress_callback` parameters and `results_to_python_dict` return type; the Python-level callback signature is unchanged, so Python callers see no difference.

No Python-level API or wheel ABI changes. `requires-python` stays at `>=3.11,<4.0`. Existing 0.9.x users on 3.11–3.13 can upgrade without code changes.

## [0.9.1] - 2026-08-08

### Documentation
- `README.md` Performance section rewritten with v0.9.0 cross-platform measurements (24 dense tests, n ≤ 300, maxdim ∈ {1, 2}, macOS arm64 + Linux x86_64 / 16-core A100); previously published numbers (2025-08) were stale.
- `README.md` adds a new section for the v0.9.0 `shuffle_null_model` FFI: 1733×–5081× median speedup vs the legacy `multiprocessing.Pool` path on a 24-cell T × N × n_shuffles grid (maxdim = 1), aggregate **484×**. Notes the semantic difference (raw spike-train Euclidean vs the full PCA + UMAP-denoise + nbs pipeline) and the auto-fallback when canns-lib lacks the FFI.
- `README.md` Citation section recommends the arXiv preprint ([arXiv:2606.27783](https://arxiv.org/abs/2606.27783)) as the primary citation, with the Zenodo archive as an optional version-specific citation. Adds an arXiv badge. (Replaces a stale dangling `benchmarks/ripiser/analysis/...` reference that no longer matches any current code path.)

### Changed
- `benchmarks/ripser/comprehensive_benchmark.py` no longer requires the legacy `canns_ripser` wheel to be installed. Adds a try/except shim that aliases `canns_lib.ripser` to `canns_ripser` when the legacy import fails, so `--fast` runs cleanly on a fresh `pip install canns-lib`. (The harness itself still compares `canns_lib.ripser.ripser` against `ripser.py`; this only changes what name the import binder resolves.)

### Infrastructure
- `.github/workflows/release.yml`: added a `concurrency: group: release-${{ github.ref }}, cancel-in-progress: false` block to prevent duplicate runs when a release tag is force-pushed shortly after the first run completes. The publish step now sets `skip-existing: true` on `pypa/gh-action-pypi-publish` so a re-run of the same tag logs "Skipping..." and exits 0 instead of failing with `400 File already exists` from PyPI.

## [0.9.0] - 2026-07-03

### Added
- New PyO3 function `canns_lib._ripser_core.shuffle_null_model(sspikes, t, n, num_shuffles, maxdim, thresh, coeff, seed)` for fast parallel shuffle null-model persistent homology. Replaces the per-shuffle `multiprocessing.Pool` + `ripser` loop used by downstream consumers (e.g., `canns` TDA shuffle analysis) with a single rayon-parallel Rust call. Supports `maxdim ∈ {0, 1, 2}`; falls through to `multiparty` callers on shape mismatch / FFI error.
- Reproducible baseline harness `benchmarks/ripser/phase_baseline.py` comparing canns-lib vs ripser.py across dense and sparse inputs, validating both **bar counts and per-dimension birth/death values** (not just counts) to catch "right bar count, wrong pairings" correctness regressions.
- Zero-apparent-pair optimisation path in `MatrixReducer` and `ColumnAssembler` (`is_in_zero_apparent_pair` + `get_zero_apparent_cofacet`/`facet`), enabling ripser-style clearing. Opt-in via env `CANNS_RIPSER_APPARENT=1`.
- Optional lock-free parallel reduction path (mod-2, no cocycles). Two correctness bugs in the prior version (filtration-rank ordering + emergent-pair shortcut) have been fixed; the path remains opt-in via `CANNS_RIPSER_USE_LOCKFREE=1` because it is currently 30× slower than the sequential reducer on this codebase (the Rips implicit-coboundary architecture doesn't materialise the full matrix the lock-free reducer would need). Left in-tree with the correctness fix for future investigation.

### Changed
- `EntryT` (and the derived `DiameterEntryT`) now pack `(index, coefficient)` into a single 64-bit word, halving the working-column memory footprint (24 → 16 bytes). This is the only micro-optimisation with measurable end-to-end impact on H1 reduction.
- `CofacetEnumerator` exposed as an associated type (GAT) on `HasCofacets`; `MatrixReducer` now uses the concrete `cofacet_enumerator` method instead of the boxed `make_enumerator`, eliminating the per-column heap allocation + vtable dispatch on the hot reduction path. Boxed `make_enumerator` remains as a default for dynamic-dispatch callers.
- `SimplexCoboundaryEnumerator` stores simplex vertices in an inline stack array (`[IndexT; 16]`) instead of a `Vec`, removing a second per-call allocation.
- `DiameterEntryT::cmp` now compares diameters by `to_bits()` instead of `f32::total_cmp`, removing NaN-aware branching from the heap comparator called on every sift.
- `Algorithm::compute_barcodes` enumerates + sorts edges once (during H0) and reuses them for the descending-order simplex list, eliminating one O(E) enumeration and one O(E log E) sort per call.
- `Phase 0/1` Python wrapper for dense distance-matrix inputs uses `np.triu_indices` instead of an O(n²) `np.meshgrid` + boolean mask per call. Now both copies of the distance matrix on the Rust side (entry) are coalesced in `MatrixReducer::new`, and the second matrix copy in `CompressedDistanceMatrix::convert_layout` is skipped when layout is already correct.
- PyO3 output no longer duplicates cocycles (`cocycles_by_dim` vs `flat_cocycles_by_dim` both returned previously) — only the flat form (the one downstream actually consumes) is returned.

### Fixed
- Lock-free parallel reducer (when enabled): ranks relabel so that "largest index = correct filtration-order pivot"; emergent-pair shortcut disabled to avoid producing right bar counts with wrong birth/death pairings (verified against ripser.py on dense and sparse inputs).

### Performance
Ripser vs ripser.py, cross-validated on **macOS arm64** (single benchmark) and **Linux x86_64 / 16-core A100 server** (LAN benchmark, `RAYON_NUM_THREADS=16`):

| | before | macOS 0.9.0 | Linux 0.9.0 |
|---|---|---|---|
| maxdim=1 median speedup | 0.53× | **0.63×** | **0.97×** |
| maxdim=2 median speedup | 0.98× | **1.10×** | **1.58×** (peak 1.74×) |
| Overall median | 0.79× | 0.79× | **1.30×** |

All 96 PyO3 Python tests + Rust unit tests pass; **counts and per-dim birth/death values match ripser.py exactly on both dense and sparse inputs** (see `benchmarks/ripser/phase_baseline.py`, which validates values not just counts to catch "right bar count, wrong pairings" correctness regressions).

The shuffle null-model FFI is typically 100-3000× faster than the equivalent `multiprocessing.Pool` path (T=100, N=40, 50 shuffles: 12 ms vs 31 s).

## [0.8.0] - 2026-06-05

### Added
- `Agent.pos` and `Agent.velocity` property setters — `agent.pos = [...]` and `agent.velocity = [...]` now go through PyO3 setters that preserve boundary projection, history sync, and head-direction normalization (PR #3)
- Top-level re-exports: `from canns_lib import Agent, Environment` no longer requires the `spatial` submodule (PR #3)
- `docs/SPATIAL_NAV_MODULE_DESIGN.md` design document, no longer gitignored (PR #3)
- Tests for property setters, rotational baseline consistency, deprecation warnings, and top-level re-exports (PR #3)

### Changed
- `Agent.set_position` and `Agent.set_velocity` are now deprecated; use the `pos` / `velocity` property setters instead. The Python wrapper emits a `DeprecationWarning`; the Rust pymethod is kept for source compatibility (PR #3)
- `Agent.set_forced_next_position` is now deprecated; use `agent.update(forced_next_position=...)` instead (PR #3)
- `python/canns_lib/spatial/__init__.py` now exposes `pos` / `position` (read-only) / `velocity` as Python-level `@property` so property assignment goes through a real setter instead of being silently shadowed by `__getattr__` (PR #3)
- `python/canns_lib/__init__.py` reads `__version__` from `importlib.metadata`; the `python/canns_lib/_version.py` shadow has been removed (PR #3)
- `Cargo.toml`: removed 8 unused dependencies (`sprs`, `indexmap`, `typed-arena`, `num-traits`, `thiserror`, dev-`approx`, dev-`criterion`, dev-`rand`); moved the `extension-module` PyO3 feature to an optional Cargo feature so `cargo test` works without flags (PR #3)
- `pyproject.toml` `[tool.maturin] features` now includes `extension-module` so `maturin develop` still builds a cdylib (PR #3)
- `example/try.py` now imports from `canns_lib.ripser` and tolerates a missing reference `ripser` install (PR #3)
- `docs/SPATIAL_NAV_MODULE_DESIGN.md` updated to match the actual source layout (`agent.rs` / `environment.rs` / `geometry.rs` / `state.rs` / `utils.rs`) (PR #3)
- `apply_set_velocity` uses `Vec::clone_from` to avoid two redundant heap allocations on each call (PR #3)
- `CLAUDE.md` documents the new `cargo test` behavior and the `extension-module` feature flag (PR #3)

### Fixed
- `set_velocity` did not update `prev_measured_velocity`, so the next `update()` computed `measured_rotational_velocity` against a stale baseline and recorded one bogus angular sample. The setter now moves the baseline in lock-step with the new velocity (PR #3)
- `cargo test --release` linker error with the `extension-module` feature enabled (PyO3 FAQ Option 1) (PR #3)
- `ZeroDivisionError` in `tests/test_complex_topology.py` sparse-vs-dense performance test (PR #2)

### Removed
- `src/ripser/ripser_old.rs` — 2,717 lines of dead, fully commented-out code (PR #3)
- `python/canns_lib/_version.py` — superseded by `importlib.metadata` (PR #3)
- `_ripser_core` and `_spatial_core` from `__all__`; they remain importable but are no longer part of the public API surface (PR #3)

## [0.7.0] - 2026-01-19

### Added
- Flexible plotting style system with three predefined styles: `simulation`, `scientific`, and `publication`
- `env.plot_environment()` method for visualizing spatial environments
- Full RatInABox API parity for agent parameter access (e.g., `agent.dt`, `agent.speed_mean`, `agent.speed_std`, `agent.rotational_velocity_std`)
- Property getters for direct access to agent configuration
- New `python/canns_lib/spatial/plotting_styles.py` module for plotting style definitions
- Example script `example/ratinabox_comparison.py` demonstrating RatInABox API compatibility (271 lines)
- Example script `example/style_comparison.py` showing all three plotting styles (110 lines)
- Test suite `tests/test_spatial_api_parity.py` for RatInABox API compatibility (77 lines)
- Comprehensive `CONTRIBUTING.md` with development setup, code style guidelines, testing procedures, and PR process (433 lines)

### Changed
- Enhanced `python/canns_lib/spatial/__init__.py` with plotting and API parity features (205 new lines)
- Updated `src/spatial/agent.rs` with property getter support (56 new lines)
- All plotting functions now support style parameter for consistent visualization

### Fixed
- Updated maintenance badge year in README.md to 2026

## [0.6.5] - 2025-XX-XX

Initial tracked release.

---

[0.8.0]: https://github.com/Routhleck/canns-lib/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Routhleck/canns-lib/compare/v0.6.5...v0.7.0
[0.6.5]: https://github.com/Routhleck/canns-lib/releases/tag/v0.6.5
