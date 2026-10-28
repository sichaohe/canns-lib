# `benchmarks/ripser`

The ripser experiment harness: the runners that produce a table, the drivers
that move the whole thing between a laptop and a shared benchmark host, and
the tools that work on the tables.

The tables themselves are **not** in this repository. They are measurements
taken on a shared host, they go stale the moment the engine changes, and they
are kept in the results project beside the analysis, not in the code.

## What may live in this repository, and what may not

> A file may live here if it is one of four things:
>
> 1. **a runner** that produces a table;
> 2. **a library** that a runner imports;
> 3. **a driver** that moves data between the laptop and the benchmark host;
> 4. **a tool** that works on tables (merges, compares, summarises) and is
>    still used.
>
> Everything else -- a probe whose question has been answered and recorded --
> **does not live here**. It moves to the process record
> (`ph_work/history/scripts/`), where "how we got here" belongs, and it is
> deleted from this repository.

The reason is not tidiness. Every file here is either something that should
be run, or something that should be read to understand why the harness is
shaped that way. A probe that answered a question in March and will never
answer it again is neither.

## The layout

```
benchmarks/ripser/
├── harness/     the runners that produce a table, and the libraries they import
├── drivers/     the shell helpers: sync, build, watch for a quiet window, fetch
├── tools/       the things that work on tables rather than produce them
├── patches/     patches applied to a COPY of ripser++, to instrument it
└── archive/     the probes that still guard something
```

A runner and the library it imports live in the same directory, so the
imports work wherever the tree is checked out.

## The rules

1. **One script produces one table.** A runner that produces two tables is
   two runners.
2. **Every table records where it came from**: the source fingerprint, the md5
   of the binary that produced it, the thread count, the load average, the
   device, and the number of cores used by processes that are **not** ours
   while that row was being measured. A number without those is not
   reproducible.
3. **A table measured on a busy host is rejected, not footnoted.** The
   benchmark host is shared, a concurrent job moves the numbers by up to 40%,
   and it can flip a comparison's sign. The gate is the **foreign CPU load**
   -- the cores used by other users' processes, read from `/proc` while the
   row was being measured -- and not the load average, which counts the
   measurement itself and therefore cannot separate our own work from
   somebody else's. A table that does not carry the reading is not a
   measurement. A second, independent line of evidence is that the same table
   is measured twice on a quiet host and the copies are compared cell by cell
   (`tools/cross_batch.py`): a neighbour's job would not make two independent
   measurements agree.
4. **The answer is checked inside the runner.** The runners compare the
   barcodes of the arms they measure, and the corpus driver compares the
   barcodes of the four canns-lib arms of every matrix. A fast wrong answer
   is worth nothing, so this cannot live in a separate program that might
   never be run.
5. **If the experiment changes the code**, the tables of *every* experiment
   are now stale: a table is only comparable with another table when the two
   came from the same build. `drivers/build_measure_sites.sh` rebuilds every
   build the tables come from, and `drivers/quiet_window_watch.sh` waits for a
   quiet window and then re-measures the chain.

## How to add an experiment

**1. Write the runner.** Import `bench_core` and use it for everything the
harness already knows how to do:

```python
import bench_core

measured = bench_core.run_measured(
    [bench_core.PY, "-c", CHILD, site, path, ...],
    env={"CANNS_RIPSER_STATS": "1"}, timeout=600, tag="my_experiment")

stats = bench_core.parse_stats(measured.stderr)      # the build's own counters
phases = bench_core.parse_phasestats(measured.stderr)  # the whole-run split
row["wall_s"] = "%.3f" % measured.wall
row["foreign_cpu"] = measured.foreign      # the quiet-window gate's number
row["verdict"] = "OK" if measured.ok else "RC=%s" % measured.rc
bench_core.write_table(out, FIELDS, rows)
```

`bench_core` gives you: the host's paths and interpreter, the
foreign-load-measured child runner, the parsers for the build's own
instrumentation, the CSV writer, the "the wrong library got imported" guard
(the editable-install trap), the measurement lock, and the build identity
(commit, source fingerprint, extension md5). **Do not re-implement any of
them.** They were re-implemented in every runner for a year and the copies
drifted.

A runner that spawns its own children takes the lock for the whole run:

```python
_LOCK = bench_core.measurement_lock(label="my_experiment")
_LOCK.__enter__()
```

The lock is an exclusive `flock`, so it is released when the process dies --
including when it is killed. The gate counts the cores used by processes that
are **not** ours, so it cannot see two of our own measurements running beside
each other, and that moves the numbers exactly as much as a neighbour's job
does.

**2. Write the CHILD script carefully.** The child is a python program held
in a string, and a name it uses but never imports fails at RUN time -- inside
a measurement, on the host, after the quiet window has been waited for. The
static check catches that class of bug in a second, before any measurement:
`tools/smoke_children.py`.

**3. Add the runner to `drivers/sync_to_server.sh`.** The file list there is
explicit on purpose: a glob would silently stop covering a file that was
renamed, and a benchmark script that did not land fails loudly while an
engine file that did not land compiles into a binary that runs and produces
numbers, which fails silently.

## The harness (`harness/`)

| runner | what it measures |
|---|---|
| `standard_bench.py` | the standard suite on ripser++'s own datasets, four sides |
| `scale_scan.py` | the scale scan, five arms, the fair comparison |
| `unified_ablation.py` | all switches on, against exactly one switch off |
| `characterize_gridcells.py` | the load the grid-cell corpus actually produces |
| `validate_gridcells.py` | cross-implementation validation on the corpus |
| `o3_profile.py` | where the time goes on the instance we lose on |
| `o3_cpu_scaling.py` | the same instance at 1, 2, 4 and 16 threads |
| `apparent_bench.py` | the apparent-pair equivalence |
| `gap_analysis.py` | same-batch gap accounting, stage by stage |
| `cofacet_early_exit_ab.py` | the cofacet early exit, two arms of one binary |
| `barcode_equivalence.py` | two builds, byte-identical barcodes required |
| `clean_h2h.py` | the head-to-head against ripser++ on ONE filtration |
| `final_h2h.py` | the final head-to-head, our GPU build against ripser++ |
| `t_arm_equivalence.py` | whether the two filtrations are the same one |
| `filtration_sweep.py` | both filtrations, both sides, one interleaved batch |
| `nsfix_ab.py` | interleaved A/B over several builds, on several arms |
| `bounds_gridcells_ab.py` | the column-store A/B on the corpus |
| `h2h_coeff_ab.py` | the coefficient field, settled by an interleaved A/B |
| `comprehensive_benchmark.py` | the comparison against `ripser.py`, the Python reference |
| `determinism_check.py` | run-to-run reproducibility of both sides |
| `cpu_baseline_giotto.py` | the CPU baseline against giotto-ph's multi-process Ripser |

The grid-cells corpus benchmark itself (`gca_bench.py`) is in the corpus
pipeline, not in this repository: the corpus' licence is still open.

### The libraries the runners import

| module | what it is for |
|---|---|
| `bench_core.py` | **the shared core.** The host's paths, the foreign-load-measured child runner, the parsers for the build's own instrumentation, the CSV writer, the wrong-library guard, the measurement lock, the build identity. A new runner imports this and writes only what is new about its experiment. |
| `foreign_cpu.py` | the number of cores another user was using while a row was measured. Read from `/proc`, so it excludes our own work by construction and needs no fudge factor for how busy we are. It is what the quiet-window gate is applied to. |
| `corpus_arms.py` | the one definition of the corpus' arms, shared by every runner that touches the corpus. |
| `apparent_fixtures.py` | the fixtures for `apparent_bench.py`. |

## The tools (`tools/`)

Run by hand: `make_ripserpp_input.py` (writes the input file for ripser++),
`merge_scale_scan.py` (merges the passes of a scale scan into one table),
`cross_batch.py` (compares two copies of the same table cell by cell -- the
second, independent line of evidence that the host was quiet),
`reduce_split.py` (our reduction against ripser++'s, split by stage),
`smoke_children.py` (the static check of the embedded child scripts),
`analysis_results.py` (turns `comprehensive_benchmark.py`'s raw output into
the plots and the summary), `summarize_characterization.py` and
`plot_characterization.py` (the characterisation table to its distributions),
`write_provenance_sidecar.py` (adds the build identity to a table that
predates the provenance columns).

## The shell helpers (`drivers/`)

| script | what it does |
|---|---|
| `sync_to_server.sh` | copies the benchmark scripts to the host, atomically, refuses an empty file, and removes the files that no longer exist here |
| `sync_engine_to_server.sh` | the same for the RUST sources, which the other one does not cover |
| `build_measure_sites.sh` | rebuilds every build the tables come from, and moves the old ones aside rather than deleting them |
| `build_and_h2h.sh` | builds the two sides and runs the head-to-head |
| `build_abl_variants.sh` | builds the ablation variants, each into its own directory |
| `quiet_window_watch.sh` | waits for a quiet window, then measures, and discards a contaminated batch |
| `measure_chain.sh` | runs the two chains, sequentially: the classical one and the corpus one |
| `fetch_clean_tables.sh` | fetches the quiet-window tables to the results project |
| `corpus_quiet_run.sh` | the same, for the corpus benchmark, whose driver is not in this repository |

## The probes (`archive/`)

Two of these still guard something, and they stay:

| script | the question it answers |
|---|---|
| `sort_gate.py` | the gate for the GPU radix sort -- the two arms have to agree, and the sorted and unsorted bar hashes have to be equal |
| `sort_neg_test.py` | its negative self-test: deliberately broken variants of the library, which the gate has to REJECT |

The three `patch_*.py` scripts are in `patches/`, not here: they are applied
to a COPY of ripser++ to instrument it, and the original binary is never
touched, because the measurements quote its time.
