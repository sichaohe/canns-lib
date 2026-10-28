# CLAUDE.md

Guidance for Claude Code when hacking on **canns-lib**.

## Project Overview

canns-lib is a single Rust crate that exports two PyO3 submodules into the Python package `canns_lib`:

- **Ripser** (`src/ripser/`) – persistent homology backend (`canns_lib._ripser_core`, API-compatible with ripser.py).
- **Spatial** (`src/spatial/`) – RatInABox-inspired spatial navigation backend (`canns_lib._spatial_core`).
- **Python layer** (`python/canns_lib/`) – re-exports both modules and keeps the version string in `_version.py`.

`src/lib.rs` registers the PyO3 package and wires both submodules into a single artifact. All Rust code lives under `src/`; there is no longer a multi-crate workspace.

## Development Commands

### Build & Install
```bash
pip install maturin
source .venv/bin/activate  # if using the repo's venv
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" maturin develop        # debug build
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" maturin develop --release
# Disable default (Rayon) feature if needed
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" maturin develop --release --no-default-features
```

### Rust Checks & Tests
```bash
cargo check                              # fast compile check
cargo fmt                                # rustfmt the tree
cargo clippy --all-targets --all-features -- -D warnings
# `extension-module` is intentionally NOT in the default feature set, so
# `cargo test` links against libpython and "just works" (PyO3 FAQ Option 1).
cargo test --release                     # runs ripser + spatial Rust tests
# To run tests with the extension-module feature enabled (e.g. for parity
# checks with maturin builds), use:
cargo test --release --features extension-module
```

### Python Tests & Benchmarks
```bash
python -m pytest tests -v
python benchmarks/compare_ripser.py --n-points 100 --maxdim 2 --trials 5
```

### Packaging
```bash
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" maturin build --release --strip --out dist
```

## Repository Layout
```
canns-lib/
├── Cargo.toml             # Single crate manifest
├── src/
│   ├── lib.rs             # Registers PyO3 package and submodules
│   ├── ripser/            # Ripser implementation
│   └── spatial/           # Spatial navigation implementation
├── python/canns_lib/
│   ├── __init__.py        # Re-export ripser, spatial, __version__
│   ├── _version.py        # Single version source
│   ├── ripser/            # Python convenience wrapper
│   └── spatial/           # Python shim for spatial backend
├── benchmarks/            # Benchmark scripts & outputs
├── docs/                  # Design docs (e.g., spatial module plan)
├── ref/                   # Reference material
└── tests/                 # Python regression tests
```

## What the work is now (2026-10-28)

**The paper is parked. The measurements are not.** The paper's drafts, its
report and the notes about them were moved to `paper-parked/` in the
experiment record (`/Volumes/home/projects/ph_work`), and the paper's
artefact layer left this repository: the nineteen checkers, the
table-to-checker map, the five report renderers, the shuffle protocol and
the three result tables that had been committed here.

What the checkers did that still has to happen now lives where the numbers
are produced: a runner compares the barcodes of the arms it measures, and
the quiet-window gate is `harness/bench_core.py`'s `foreign_summary` and
`batch_is_clean`, which every driver and the fetch script call.

**The goal is the engine's performance, measured across benchmarks** --
the standard suite, the scale scan, the head-to-head against ripser++ and
the grid-cells corpus, which is our own benchmark and the one that matters
most. A change is judged by what it does to those tables, not by whether it
supports a sentence.

## Implementation Notes

### Python Layer
- `python/canns_lib/__init__.py` exposes `ripser`, `spatial`, and `__version__`.
- `python/canns_lib/ripser/__init__.py` mirrors ripser.py.
- `python/canns_lib/spatial/__init__.py` forwards to the Rust extension and warns if it is missing.

### Rust Layer
- `src/lib.rs` defines the `canns_lib` PyModule and registers `_ripser_core` and `_spatial_core` via helper functions.
- `src/ripser/mod.rs` exports `ripser_dm`, `ripser_dm_sparse`, etc.
- `src/spatial/mod.rs` implements `Environment`/`Agent` with geometry helpers, OU dynamics, history export, and unit tests.

## Feature Flags
- Default build enables Rayon (`parallel`). Use `--no-default-features` if running in constrained environments.
- `extension-module` (Cargo feature, off by default) makes PyO3 build a cdylib suitable for `import` from Python. Maturin enables it automatically via `[tool.maturin] features` in `pyproject.toml`. Enabling it during a plain `cargo test` run will break the link on macOS/Linux because the resulting binary does not pull in `libpython`.

## Testing Strategy
- Keep Python tests in `tests/` aligned with ripser.py and RatInABox expectations; seed randomness for determinism.
- Add Rust unit/property tests in `src/ripser/` and `src/spatial/` for geometric and stochastic behaviour.

## Contribution Tips
- Maintain RatInABox API parity in the spatial module (method signatures, parameter names, history format).
- Document new env vars or build requirements in `README.md` / `docs/`.
- Provide updated benchmarks or screenshots for performance/UI changes.

## Repository Hygiene

- **Every change carries its own cleanup.** A commit that leaves a dead script,
  a superseded copy or a stale table behind is not finished. Delete it in the
  same commit, or say in the commit message why it stays.
- **The benchmark tree has a written rule for what may live in it.**
  `benchmarks/ripser/README.md` states it: a file there has to be a runner, a
  library, a driver, a tool or a patch. A probe whose question is answered
  does not qualify -- it goes to the paper project's process record
  (`ph_work/history/scripts/`) and is deleted from here.
- **Do not re-implement what `benchmarks/ripser/harness/bench_core.py` already
  does.** It holds the host's paths, the foreign-load-measured child runner,
  the parsers for the build's own instrumentation, the CSV writer, the
  wrong-library guard and the build identity. They were re-implemented in
  every runner for a year and the copies drifted.
- **Work on a branch.** `master` carries released work; a change goes on a
  branch and lands through a PR.
- **A measurement script's semantics do not change.** The same input must
  produce the same table. If a refactor is meant to preserve the output, say
  how that was checked; if it is meant to change the output, say what changed
  and re-measure every table that depended on it.
