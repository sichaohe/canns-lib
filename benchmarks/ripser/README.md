# `benchmarks/ripser`

The ripser benchmark harness: the runners that produce a table, the checkers
that decide whether the table is any good, and the probes that answered a
question once and are kept for the record.

The tables themselves are **not** in this repository. Two of them are computed
from a grid-cell corpus whose licence is not settled, and a table that names
every recording of the corpus is more than "the scripts and the aggregate
characterisation". The scripts, the checkers and the aggregate characterisation
are here; the per-matrix tables live in the artefact, and will move here when
the licence question is answered.

## The layout

```
benchmarks/ripser/
├── harness/     the runners that produce a table, and the two libraries they import
├── checkers/    one checker per table
├── drivers/     the shell helpers: sync, fetch, watch for a quiet window, build
├── tools/       the things that work on tables rather than produce them
├── patches/     patches applied to a COPY of ripser++, to instrument it
├── archive/     the probes that answered a question once
└── (the rest)   what came with the repository, and three tables
```

A runner and the library it imports live in the same directory, so the
imports work wherever the tree is checked out; the checkers import nothing
from each other, so they can be read one at a time.

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
   row was measured -- and not the load average, which counts the
   measurement itself and therefore cannot separate our own work from
   somebody else's. The checkers take `--max-foreign-cpu`, and the watcher
   that produces the tables discards a whole batch rather than keeping half
   of it. A second, independent line of evidence is that the same table is
   measured twice on a quiet host and the copies are compared cell by cell
   (`tools/cross_batch.py`): a neighbour's job would not make two independent
   measurements agree.

## The harness (`harness/`)

| runner | what it measures | checker |
|---|---|---|
| `clean_h2h.py` | the head-to-head against ripser++ on ONE filtration | `check_clean_h2h.py` |
| `final_h2h.py` | the final head-to-head, our GPU build against ripser++ | `check_final_h2h.py` |
| `standard_bench.py` | the standard suite on ripser++'s own datasets, four sides | `check_standard_bench.py` |
| `characterize_gridcells.py` | the load the grid-cell corpus actually produces | `check_characterize_gridcells.py` |
| `scale_scan.py` | the scale scan, five arms, the fair comparison | `check_scale_scan.py` |
| `unified_ablation.py` | all switches on, against exactly one switch off | `check_unified_ablation.py` |
| `t_arm_equivalence.py` | whether the two filtrations are the same one | `check_t_arm_equivalence.py` |
| `validate_gridcells.py` | cross-implementation validation on the corpus | `check_validate_gridcells.py` |
| `filtration_sweep.py` | both filtrations, both sides, one interleaved batch | `check_filtration_sweep.py` |
| `nsfix_ab.py` | interleaved A/B over several builds, on several arms | `check_nsfix_ab.py` |
| `bounds_gridcells_ab.py` | the column-store A/B on the corpus | `check_bounds_ab.py` |
| `apparent_bench.py` | the apparent-pair equivalence | `check_apparent_bench.py` |
| `gap_analysis.py` | same-batch gap accounting, stage by stage | `check_gap.py` |
| `h2h_coeff_ab.py` | the coefficient field, settled by an interleaved A/B | `check_h2h_coeff_ab.py` |
| `o3_profile.py` | where the time goes on the instance we lose on | `check_o3_profile.py` |
| `o3_cpu_scaling.py` | the same instance at 1, 2, 4 and 16 threads | `check_o3_cpu_scaling.py` |
| `determinism_check.py` | run-to-run reproducibility of both sides | -- |
| `cpu_baseline_giotto.py` | the CPU baseline against giotto-ph's multi-process Ripser | -- |
| `gca_bench.py` (in the corpus, not in this repository) | the grid-cells main benchmark | `check_gridcells_bench.py` |

Every checker lives in `checkers/`, with the same name as the runner it
guards. `check_reduce_split.py` is the one that has no runner of its own: it
guards the table that compares the two implementations' reduction, and it
is the check that makes "both sides do the same work" a fact rather than a
claim.

Called by the runners: `h2h_child.py` (one measured run of our ripser, used
by `final_h2h.py`, `gap_analysis.py` and the archived probes) and
`apparent_fixtures.py` (the fixtures for `apparent_bench.py`).

Called by every runner that measures time: `foreign_cpu.py`. It reads every
process's CPU time from `/proc`, sums the ones owned by another user, and
hands the runner the number of cores that somebody else was using while
the row was measured. It is what the quiet-window gate is applied to.

`cpu_baseline_giotto.py` has no checker because it could not be run: giotto-ph
does not install in the benchmark environment (its wheel unpacks to a
dist-info only, and `import giotto` fails). The table it writes records the
failure in the `giotto_error` column rather than an empty number.

## The tools (`tools/`)

Run by hand: `make_ripserpp_input.py` (writes the input file for ripser++),
`merge_scale_scan.py` (merges the passes of a scale scan into one table),
`cross_batch.py` (compares two copies of the same table cell by cell -- the
second, independent line of evidence that the host was quiet) and
`reduce_split.py` (our reduction against ripser++'s, split by stage).

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
| `sync_to_server.sh` | copies the source to the benchmark host, atomically, and refuses an empty file |
| `build_and_h2h.sh` | builds the two sides and runs the head-to-head |
| `build_abl_variants.sh` | builds the ablation variants, each into its own directory |
| `quiet_window_watch.sh` | waits for a quiet window, then measures, and discards a contaminated batch |
| `fetch_clean_tables.sh` | fetches the quiet-window tables and checks each one |
| `corpus_quiet_run.sh` | the same, for the corpus benchmark, whose driver is not in this repository |

## The probes (`archive/`)

These answered a question once. They are kept because the answer is in the
papers and the report, and because a claim that cannot be re-run is a claim
that cannot be checked. They are not tables, and most of them have no
checker.

| script | the question it answered |
|---|---|
| `inf_form_probe.py` | what exactly differs between the `+inf`-preserved and the capped input |
| `thresh_probe.py` | whether the 30% `+inf` in the ASA matrix is actually costing us |
| `loop_probe.py` | how the reducing columns are actually reduced, from the counters |
| `sparse_paths.py` | three ways to read the same 30%-`+inf` matrix |
| `sort_gate.py`, `sort_neg_test.py` | the gate for the GPU radix sort, and its negative self-test |
| `two_site_ab.py`, `onepass_ab.py`, `pinned_cache_ab.py` | A/Bs between two builds, in one interleaved batch |
| `h2h_child.py` | one measured run of our ripser, used by the probes |
| `p21_gate.sh` | the gate for the auto-sparse switch. **Archived because it is broken**: it calls `compare_p4.py`, which does not exist in this repository, so it cannot run |

The three `patch_*.py` scripts are in `patches/`, not here: they are applied
to a COPY of ripser++ to instrument it, and they are used by the probes and
by the reduction comparison, so they are kept where they can be run.

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

`ASA_SHUFFLE.md`, `analysis/`, `analysis_results.py`, `comprehensive_benchmark.py`,
`phase_baseline.py` and `progress_bar_impact.py` predate this work.
