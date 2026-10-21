"""The quiet-window gate: one implementation, used by every checker.

WHY THIS IS A SEPARATE FILE. The gate is the difference between a table that
is a measurement of our code and a table that is a measurement of the host.
The benchmark host is shared, a neighbour's job moves the numbers by up to
40%, and it can flip a comparison's sign -- so a table measured on a busy
host is not a table with a caveat, it is not a table.

Seven checkers had the gate and twelve did not, which meant that twelve
tables could not be shown to have been measured on a quiet host at all. The
gate is now here, and every checker that guards a timing table calls it.

THE NUMBER IS THE FOREIGN LOAD, NOT THE LOAD AVERAGE. The load average
counts our own measurement: the reduction runs on the host with one thread
per core, which puts the load average at 16 on this 16-core box, and a
neighbour's two-core job then hides underneath any threshold that clears our
own footprint. The `foreign_cpu` column is the number of cores used by
processes that are NOT ours, read from `/proc` while the row was being
measured, so ours is excluded by construction and the number needs no
allowance for how busy the experiment is. The peak (`foreign_max`) is a
DIAGNOSTIC: a half-second blip from the box's own desktop session is not a
reason to throw away a two-minute measurement.

WHAT A TABLE WITHOUT THE COLUMN GETS. A failure, and a specific one: it
cannot be shown to have been measured on a quiet host. That is not a
technicality -- it is the whole claim. A table that predates the column has
to be re-measured, not excused.

A checker for a table whose numbers do NOT depend on the host's speed --
a comparison of bar values, of counts, of a fingerprint -- does not call
this at all, and says so in its docstring.
"""
from __future__ import annotations

import csv
import os
import sys

# The threshold every checker defaults to, and the one the fetch script
# passes. The number in the flag and the number in the checker have to be
# the same number.
DEFAULT_GATE = 0.25


def load_rows(path):
    """The rows of a table, or [] if it cannot be read."""
    try:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def gate(table, max_foreign_cpu=DEFAULT_GATE, label=""):
    """The failures of a table with respect to the quiet window.

    `table` is a path or a list of rows. Returns a list of failure strings
    (empty == the table is quiet enough to be a measurement).
    """
    if max_foreign_cpu <= 0:
        return []
    rows = load_rows(table) if isinstance(table, str) else list(table)
    who = label or (table if isinstance(table, str) else "the table")
    if not rows:
        return ["%s: the table is empty, so there is nothing to check" % who]
    if "foreign_cpu" not in rows[0]:
        return ["%s: the table carries no foreign-load reading, so it cannot be "
                "shown to have been measured on a quiet host -- re-measure it "
                "with a runner that records the foreign load" % who]
    worst_mean, worst_max, bad = 0.0, 0.0, None
    for r in rows:
        raw = (r.get("foreign_cpu") or "").strip()
        if not raw:
            bad = "%s: a row has no foreign-load reading" % who
            break
        try:
            fc = float(raw)
        except ValueError:
            bad = "%s: the foreign-load column is not a number (%r)" % (who, raw)
            break
        try:
            fm = float((r.get("foreign_max") or "").strip() or 0.0)
        except ValueError:
            fm = 0.0
        worst_mean = max(worst_mean, fc)
        worst_max = max(worst_max, fm)
        if fc > max_foreign_cpu and bad is None:
            bad = ("%s: the foreign load %s cores on a row is above the "
                   "quiet-window threshold %.2f -- somebody else was using the "
                   "host while that row was measured"
                   % (who, raw, max_foreign_cpu))
    print("worst foreign load: mean %.3f cores, largest single window %.3f "
          "cores (threshold %.2f)" % (worst_mean, worst_max, max_foreign_cpu))
    return [bad] if bad else []


def add_arguments(ap, default=DEFAULT_GATE):
    """The `--max-foreign-cpu` flag, for a checker's own parser."""
    ap.add_argument("--max-foreign-cpu", type=float, default=default,
                    help="the quiet-window gate: the mean foreign cores a row "
                         "may carry. 0 disables the gate, and is only right "
                         "for a table whose numbers do not depend on the "
                         "host's speed")
    return ap
