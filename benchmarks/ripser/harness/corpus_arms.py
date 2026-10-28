"""The arms of the grid-cells corpus table, in ONE place.

WHY THIS FILE EXISTS. The arm list lived in two places -- the corpus'
own driver (which is not in either repository, because the corpus pipeline's
licence is still open) and the checker in this repository -- and the two
copies were coupled by hand. That is how the corpus driver ended up looking
for `foreign_cpu` in the tree root after the harness became a tree: the
driver imports from this repository, so a MOVE here breaks it, and the
coupling is invisible from either side. This file is the one definition, and
both the driver and the checker read it.

THE THRESHOLD IS A TABLE-LEVEL CONSTANT, NOT A PER-ARM PARAMETER.

Every arm computes the SAME filtration: the one where the genuinely missing
pairs are excluded. That is what makes a ratio between two arms a statement
about the implementations rather than about the problems.

* the maximum finite distance of the matrix, `T = max(finite)`, excludes
  exactly the pairs that the `+inf` representation keeps as missing;
* the `+inf` representation with the threshold at `T` therefore has the same
  edges as the `capped` representation with the threshold at `T`: the
  capped pairs sit at `2*max_finite + 1`, which is above `T`, so they are
  excluded on both sides.

So `T` is the threshold of every arm, and the input form is only the
REPRESENTATION of the same matrix.

What the arms are for:

| arm | implementation | input form | what it isolates |
|---|---|---|---|
| `canns-gpu` | ours, GPU | `+inf` | the headline |
| `canns-cpu` | ours, CPU | `+inf` | what the device buys |
| `canns-gpu` | ours, GPU | `capped` | the cost of the representation |
| `canns-cpu` | ours, CPU | `capped` | the same, on the CPU build |
| `ripserpp_T` | ripser++ | `capped` | the denominator |

The arm with ripser++ at the CAP threshold is NOT here on purpose: it admits
every pair in the file as a real edge, so it computes a DIFFERENT filtration.
It is a legitimate measurement of the corpus' own default configuration, but
using it as a denominator compares two different problems, and it flatters
us: the corpus table's 0.641x against that arm is not a claim about the same
work. The corpus' own benchmark keeps that arm in its own table.
"""
from __future__ import annotations

# The threshold of every arm. `T` is the maximum finite distance of the
# matrix; the driver computes it per matrix.
THRESHOLD = "T"

# (target, input_form)
ARMS = [
    ("canns-gpu", "inf_preserved"),
    ("canns-cpu", "inf_preserved"),
    ("canns-gpu", "capped"),
    ("canns-cpu", "capped"),
    ("ripserpp_T", "capped"),
]

# The arms that are OURS: the ones that must agree with each other on the
# barcode, and the ones whose times are compared with the reference's.
OUR_ARMS = [a for a in ARMS if a[0].startswith("canns-")]
REFERENCE_ARMS = [a for a in ARMS if not a[0].startswith("canns-")]


def cells(n_matrices):
    """How many cells the table has."""
    return n_matrices * len(ARMS)
