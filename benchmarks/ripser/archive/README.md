# `benchmarks/ripser/archive/`

The probes that **still guard something**. They are not runners and they do
not produce a table, so they are not in `harness/`.

| script | what it guards |
|---|---|
| `sort_gate.py` | the gate for the GPU radix sort -- the two arms have to agree, and the sorted and unsorted bar hashes have to be equal |
| `sort_neg_test.py` | its negative self-test: deliberately broken variants of the library, which the gate has to REJECT |

If the sort changes, these are the two to re-run.

## What used to be here

This directory used to hold eleven probes that answered a question once and
guarded nothing afterwards. They are not here any more, and the rule that
moved them is in the parent README ("What may live in this repository"): a
file in this repository has to be a runner, a library, a driver, a tool or a
patch. A probe whose question is answered does not qualify.

They were not deleted. They are in the paper project's process record
(`ph_work/history/scripts/`), together with the other one-off scripts from
the same period, and in this repository's git history:

| script | the question it answered |
|---|---|
| `inf_form_probe.py` | what exactly differs between the `+inf`-preserved and the capped input |
| `thresh_probe.py` | whether the 30% `+inf` in the ASA matrix is actually costing us |
| `loop_probe.py` | how the reducing columns are actually reduced, from the counters |
| `sparse_paths.py` | three ways to read the same 30%-`+inf` matrix |
| `two_site_ab.py` | A/B between two builds, in one interleaved batch |
| `onepass_ab.py` | the same, for the one-pass variant |
| `pinned_cache_ab.py` | the same, with the pinned-memory cache |

Two more were removed outright, because they were not "kept for the record"
but simply dead:

* `h2h_child.py` -- a **copy** of a file that lives in the paper project. The
  runners that used it pointed at `/home/sichaohe/ph_work/h2h_child.py`, not
  at this one, so nothing ever ran this one.
* `p21_gate.sh` -- **broken**: it calls `compare_p4.py`, which does not exist
  in this repository, so it cannot run at all. A gate that cannot run is
  not a gate; it is a file that looks like one.
