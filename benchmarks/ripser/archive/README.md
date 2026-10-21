# `benchmarks/ripser/archive/`

The probes: scripts that answered a question once and are kept because the
answer is in the papers and the report, and because a claim that cannot be
re-run is a claim that cannot be checked. They are not tables, and most of
them have no checker.

| script | the question it answered |
|---|---|
| `inf_form_probe.py` | what exactly differs between the `+inf`-preserved and the capped input |
| `thresh_probe.py` | whether the 30% `+inf` in the ASA matrix is actually costing us |
| `loop_probe.py` | how the reducing columns are actually reduced, from the counters |
| `sparse_paths.py` | three ways to read the same 30%-`+inf` matrix |
| `sort_gate.py` | the gate for the GPU radix sort |
| `sort_neg_test.py` | its negative self-test |
| `two_site_ab.py` | A/B between two builds, in one interleaved batch |
| `onepass_ab.py` | the same, for the one-pass variant |
| `pinned_cache_ab.py` | the same, with the pinned-memory cache |
| `h2h_child.py` | one measured run of our ripser, used by the probes and by `gap_analysis.py` |
| `p21_gate.sh` | the gate for the auto-sparse switch -- **archived because it is broken**: it calls `compare_p4.py`, which does not exist in this repository, so it cannot run |

Two of these (`sort_gate.py`, `sort_neg_test.py`) still guard something: the
GPU radix sort. If the sort changes, they are the ones to re-run, and they
belong here rather than in `checkers/` because they do not guard a table.
