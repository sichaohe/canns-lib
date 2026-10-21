# `benchmarks/ripser`

The ripser benchmark harness: the runners that produce a table, the checkers
that decide whether the table is any good, and the drivers that move the
whole thing between a laptop and a shared benchmark host.

The tables themselves are **not** in this repository. Two of them are computed
from a grid-cell corpus whose licence is not settled, and a table that names
every recording of the corpus is more than "the scripts and the aggregate
characterisation". The scripts, the checkers and the aggregate characterisation
are here; the per-matrix tables live in the artefact, and will move here when
the licence question is answered.

## What may live in this repository, and what may not

This used to be decided by history, which is how a repository ends up with
eighty-one files in one directory and no way to tell which of them still
matter. It is decided by this rule now:

> A file may live here if it is one of five things:
>
> 1. **a runner** that produces a table the paper quotes, and that table has
>    a checker;
> 2. **a library** that a runner imports;
> 3. **a driver** that moves data between the laptop and the benchmark host;
> 4. **a tool** that works on tables (renders, merges, compares) and is still
>    used;
> 5. **a patch** applied to a *copy* of a reference implementation, still
>    used.
>
> Everything else -- a probe whose question has been answered and recorded --
> **does not live here**. It moves to the paper project's process record
> (`ph_work/history/scripts/`), where "how we got here" belongs, and it is
> deleted from this repository.

The reason is not tidiness. This repository is the paper's **artefact**: a
reader who comes here to reproduce a number has to be able to tell what is
part of the method and what is a leftover. Every file here is either
something they should run, or something they should read to understand why
the method is shaped that way. A probe that answered a question in March and
will never answer it again is neither, and a reader who has to work out which
is which has been handed the wrong repository.

The process record is not lost. It is in the paper project, and it is in this
repository's git history.

## The layout

```
benchmarks/ripser/
├── harness/     the runners that produce a table, and the libraries they import
├── checkers/    one checker per table
├── drivers/     the shell helpers: sync, build, watch for a quiet window, fetch
├── tools/       the things that work on tables rather than produce them
├── patches/     patches applied to a COPY of ripser++, to instrument it
├── archive/     the probes that still guard something
└── (the rest)   what came with the repository, and three tables
```

A runner and the library it imports live in the same directory, so the
imports work wherever the tree is checked out; the checkers import nothing
from each other, so they can still be read one at a time.

## The rules

1. **One script produces one table, one checker checks one table.** A table
   without a checker is a diagnostic, not a result.
2. **The checker derives its grid from the table.** No cell count is written
   down twice, so a missing cell is a failure and not a silent hole.
3. **Every table records where it came from**: the source fingerprint, the md5
   of the binary that produced it, the thread count, the load average, the
   device, and the number of cores used by processes that are **not** ours
   while that row was being measured. A number without those is not
   reproducible.
4. **Every checker has negative self-tests.** They mutate a copy of the table
   and require the checker to notice. A checker that cannot fail is not a
   checker; the self-tests are what make that claim checkable.
5. **A table measured on a busy host is rejected, not footnoted.** The
   benchmark host is shared, a concurrent job moves the numbers by up to 40%,
   and it can flip a comparison's sign. The gate is the **foreign CPU load**
   -- the cores used by other users' processes, read from `/proc` while the
   row was being measured -- and not the load average, which counts the
   measurement itself and therefore cannot separate our own work from
   somebody else's. The checkers take `--max-foreign-cpu`, and the watcher
   that produces the tables discards a whole batch rather than keeping half
   of it. A second, independent line of evidence is that the same table is
   measured twice on a quiet host and the copies are compared cell by cell
   (`tools/cross_batch.py`): a neighbour's job would not make two independent
   measurements agree.

## How to add an experiment

This is the part that is supposed to make the tree reusable. A new
measurement is four steps, and the first three are the same for every
experiment, so they live in one place.

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
(the editable-install trap), and the build identity (commit, source
fingerprint, extension md5). **Do not re-implement any of them.** They were
re-implemented in every runner for a year and the copies drifted.

**2. Write the checker**, in `checkers/`, named after the runner. It must
derive its grid from the table, take `--max-foreign-cpu`, and carry negative
self-tests. `check_cofacet_early_exit_ab.py` is the smallest complete example
of the shape.

**3. Add the runner and the checker to `drivers/sync_to_server.sh`.** The file
list there is explicit on purpose: a glob would silently stop covering a file
that was renamed, and a benchmark script that did not land fails loudly while
an engine file that did not land compiles into a binary that runs and
produces numbers, which fails silently.

**4. Add the table to `RESULTS-20261020.md`**, which is the map from a table
to the checker that guards it, and which is where a reader should start.

**5. If the experiment changes the code**, the tables of *every* experiment
are now stale: a table is only comparable with another table when the two
came from the same build. `drivers/build_measure_sites.sh` rebuilds every
build the tables come from, and `drivers/quiet_window_watch.sh` waits for a
quiet window and then re-measures the chain.

## The harness (`harness/`)

| runner | what it measures | checker |
|---|---|---|
| `standard_bench.py` | the standard suite on ripser++'s own datasets, four sides | `check_standard_bench.py` |
| `scale_scan.py` | the scale scan, five arms, the fair comparison | `check_scale_scan.py` |
| `unified_ablation.py` | all switches on, against exactly one switch off | `check_unified_ablation.py` |
| `characterize_gridcells.py` | the load the grid-cell corpus actually produces | `check_characterize_gridcells.py` |
| `validate_gridcells.py` | cross-implementation validation on the corpus | `check_validate_gridcells.py` |
| `o3_profile.py` | where the time goes on the instance we lose on | `check_o3_profile.py` |
| `o3_cpu_scaling.py` | the same instance at 1, 2, 4 and 16 threads | `check_o3_cpu_scaling.py` |
| `apparent_bench.py` | the apparent-pair equivalence | `check_apparent_bench.py` |
| `gap_analysis.py` | same-batch gap accounting, stage by stage | `check_gap.py` |
| `cofacet_early_exit_ab.py` | the cofacet early exit, two arms of one binary | `check_cofacet_early_exit_ab.py` |
| `barcode_equivalence.py` | two builds, byte-identical barcodes required | -- |
| `clean_h2h.py` | the head-to-head against ripser++ on ONE filtration | `check_clean_h2h.py` |
| `final_h2h.py` | the final head-to-head, our GPU build against ripser++ | `check_final_h2h.py` |
| `t_arm_equivalence.py` | whether the two filtrations are the same one | `check_t_arm_equivalence.py` |
| `filtration_sweep.py` | both filtrations, both sides, one interleaved batch | `check_filtration_sweep.py` |
| `nsfix_ab.py` | interleaved A/B over several builds, on several arms | `check_nsfix_ab.py` |
| `bounds_gridcells_ab.py` | the column-store A/B on the corpus | `check_bounds_ab.py` |
| `h2h_coeff_ab.py` | the coefficient field, settled by an interleaved A/B | `check_h2h_coeff_ab.py` |
| `comprehensive_benchmark.py` | the comparison against `ripser.py`, the Python reference | -- |
| `determinism_check.py` | run-to-run reproducibility of both sides | -- |
| `cpu_baseline_giotto.py` | the CPU baseline against giotto-ph's multi-process Ripser | -- |
| `gca_bench.py` (in the corpus, not in this repository) | the grid-cells main benchmark | `check_gridcells_bench.py` |

Every checker lives in `checkers/`, with the same name as the runner it
guards. `check_reduce_split.py` is the one that has no runner of its own: it
guards the table that compares the two implementations' reduction, and it
is the check that makes "both sides do the same work" a fact rather than a
claim.

### The libraries the runners import

| module | what it is for |
|---|---|
| `bench_core.py` | **the shared core.** The host's paths, the foreign-load-measured child runner, the parsers for the build's own instrumentation, the CSV writer, the wrong-library guard, the build identity. A new runner imports this and writes only what is new about its experiment. |
| `foreign_cpu.py` | the number of cores another user was using while a row was measured. Read from `/proc`, so it excludes our own work by construction and needs no fudge factor for how busy we are. It is what the quiet-window gate is applied to. |
| `apparent_fixtures.py` | the fixtures for `apparent_bench.py`. |

`cpu_baseline_giotto.py` has no checker because it could not be run: giotto-ph
does not install in the benchmark environment (its wheel unpacks to a
dist-info only, and `import giotto` fails). The table it writes records the
failure in the `giotto_error` column rather than an empty number.

## The tools (`tools/`)

Run by hand: `make_ripserpp_input.py` (writes the input file for ripser++),
`merge_scale_scan.py` (merges the passes of a scale scan into one table),
`cross_batch.py` (compares two copies of the same table cell by cell -- the
second, independent line of evidence that the host was quiet),
`reduce_split.py` (our reduction against ripser++'s, split by stage) and
`analysis_results.py` (turns `comprehensive_benchmark.py`'s raw output into
the plots and the summary).

## Rendering the paper's sections from the tables (`tools/`)

`render_characterization_section.py` writes the load-characterisation section
from the characterisation table; `summarize_characterization.py` reduces that
table to the distributions the paper quotes; `plot_characterization.py` draws
the two figures; `render_clean_md.py`, `render_final_md.py`,
`render_gap_md.py` and `render_ripser_report.py` write the reports.
`write_provenance_sidecar.py` adds the build identity to a table that predates
the provenance columns.

## The shell helpers (`drivers/`)

| script | what it does |
|---|---|
| `sync_to_server.sh` | copies the benchmark scripts to the host, atomically, and refuses an empty file |
| `sync_engine_to_server.sh` | the same for the RUST sources, which the other one does not cover |
| `build_measure_sites.sh` | rebuilds every build the tables come from, and moves the old ones aside rather than deleting them |
| `build_and_h2h.sh` | builds the two sides and runs the head-to-head |
| `build_abl_variants.sh` | builds the ablation variants, each into its own directory |
| `quiet_window_watch.sh` | waits for a quiet window, then measures, and discards a contaminated batch |
| `fetch_clean_tables.sh` | fetches the quiet-window tables and checks each one |
| `corpus_quiet_run.sh` | the same, for the corpus benchmark, whose driver is not in this repository |

## The probes (`archive/`)

Two of these still guard something, and they stay:

| script | the question it answers |
|---|---|
| `sort_gate.py` | the gate for the GPU radix sort -- the two arms have to agree, and the sorted and unsorted bar hashes have to be equal |
| `sort_neg_test.py` | its negative self-test: deliberately broken variants of the library, which the gate has to REJECT |

The probes that answered a question once and guard nothing now are in the
paper project's process record, not here. See "What may live in this
repository" above for the rule that decided it.

The three `patch_*.py` scripts are in `patches/`, not here: they are applied
to a COPY of ripser++ to instrument it, and the original binary is never
touched, because the paper quotes its time.

## The paper's drafts

**The drafts are not in this repository.** It is public, and a draft paper
is not part of the artefact: this repository carries the harness and the
map between table and checker, which is what a reader needs in order to
reproduce the numbers.

`RESULTS-20261020.md` maps each table to its checker, and it is the file
a reader of the paper should start from.

The drafts (the story, the abstract, the introduction, the
load-characterisation section generated by
`tools/render_characterization_section.py`, and the tables they are views
of) live in a separate project repository, beside the raw tables
themselves.

## What came with the repository

`ASA_SHUFFLE.md`, `comprehensive_benchmark.py` (now in `harness/`) and
`analysis_results.py` (now in `tools/`) predate this work. The first two
measure against `ripser.py`, the Python reference, and the repository's own
README attributes its headline performance numbers to them -- so they are
part of the method, and they are kept. The one PNG in `analysis/` was a
sample output of `analysis_results.py` and was removed.
