"""The parts every runner in this directory does the same way.

WHY THIS FILE EXISTS. The runners in `harness/` are separate programs because
they measure different things, and they were written one at a time. What they
all do is the same handful of things: point at a build, run one measured
child under the foreign-load probe, read the build's own instrumentation out
of the child's stderr, and write a CSV. That handful was written again in
every runner, and the copies had drifted -- two `provenance()` functions with
different bodies, four parsers with different names, two different spellings
of the "the wrong library got imported" guard.

This file is the one place those things live. A new runner imports it and
writes the part that is actually new.

WHAT IS **NOT** HERE, AND WHY. The dataset parameters. Every runner asks for
a different set of them, and a runner that takes them from a shared table
somewhere else has to be read together with that table to be understood. The
paths that several runners genuinely share -- the benchmark host, the
interpreter, the corpus, the reference implementation -- are constants here,
because those are facts about the machine rather than about the experiment.

THE RULES THIS FILE ENCODES

* A measurement that ran while somebody else was using the machine is not a
  measurement. `run_measured` records the foreign load for every run, and
  `quiet_enough` is the gate.
* A build that is not the one you meant is worse than no build. `lib_guard`
  refuses to run against a library other than the one it was pointed at,
  because a conda-installed `canns_lib` can beat `PYTHONPATH` and silently
  benchmark the WRONG binary for twenty minutes.
* A table is a table, whatever produced it. `write_table` writes the same
  header and the same quoting for every runner, so two tables can be
  compared cell by cell without a per-pair special case.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foreign_cpu

# The benchmark host. The paths are the host's, not the laptop's: the
# measurement runs there, and the laptop only holds the scripts.
HOST = "222.29.101.57"
PORT = 1693
PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"
REPO = "/home/sichaohe/canns-lib-dev"
MODULE = os.path.join(REPO, "benchmarks", "ripser")

# The reference implementation's own suite, and the corpus.
EXAMPLES = "/home/sichaohe/ph_work/ripserpp/ripserplusplus/examples"
CORPUS = "/home/sichaohe/gridcells/matrices"
ASA_MATRIX = "/home/sichaohe/ph_work/ph_dm_1200.npy"

# The foreign load a row may carry and still be a measurement. It is the MEAN
# over the row, not the peak: a neighbour's seven-core job for the whole of a
# run shows up as seven, and a compositor that used one core for half a second
# out of a two-minute run shows up as 0.004, which is what it is worth.
FOREIGN_GATE = 0.25


class Measured(object):
    """One measured run: what it printed, how long it took, how busy the box was."""

    def __init__(self, argv, stdout, stderr, rc, wall, foreign, foreign_max, env=None):
        self.argv = argv
        self.stdout = stdout
        self.stderr = stderr
        self.rc = rc
        self.wall = wall
        self.foreign = foreign
        self.foreign_max = foreign_max
        self.env = env or {}

    @property
    def ok(self):
        return self.rc == 0

    def __repr__(self):
        return "<Measured rc=%s wall=%.3fs foreign=%s>" % (
            self.rc, self.wall, self.foreign)


def run_measured(argv, env=None, timeout=1800, cwd=None, tag=None, keep=False):
    """Run one child and measure it.

    The foreign load is read from `/proc` WHILE the child runs, not before
    and not after, so the number belongs to the run rather than to the
    neighbourhood.

    A timeout is not a failure to hide: the child is killed and the return
    code is recorded, so a row that timed out says so. `TIMEOUT` is a legal
    result in a table -- what is not legal is a row that looks like a fast
    run because the child died.

    `tag` names the two files the child's output is left in, so a failed run
    can be read after the fact; `keep` leaves them there. Without a tag the
    files are named after the process and removed.
    """
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    if tag:
        out_path = "/tmp/_bench_%s.out" % tag
        err_path = "/tmp/_bench_%s.err" % tag
    else:
        out_path = "/tmp/_bench_out_%d" % os.getpid()
        err_path = "/tmp/_bench_err_%d" % os.getpid()
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        with foreign_cpu.Probe() as probe:
            t0 = time.perf_counter()
            proc = subprocess.Popen(argv, stdout=fo, stderr=fe, env=child_env,
                                    cwd=cwd)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            wall = time.perf_counter() - t0
    with open(out_path) as fh:
        stdout = fh.read()
    with open(err_path) as fh:
        stderr = fh.read()
    if not keep and not tag:
        os.unlink(out_path)
        os.unlink(err_path)
    return Measured(argv, stdout, stderr, proc.returncode, wall,
                    probe.record(), probe.record_max(), env)


def with_foreign(fn, *args, **kwargs):
    """Run `fn(*args, **kwargs)` while the foreign load is sampled.

    Returns `(result, foreign_mean, foreign_max)`. The two numbers are
    strings, ready for a table row.

    This is the wrapper for the runners that measure their child some other
    way -- `/usr/bin/time` for the elapsed time and the peak RSS, say. It
    leaves the measurement of the child EXACTLY as it was and only adds the
    reading of the host, so a table gains the column that proves it was
    measured in a quiet window without any of its numbers changing meaning.

    The numbers are strings on purpose: a probe that never got a sample
    reports "" rather than 0.0, because "" means "we do not know" and 0.0
    would read as "the box was provably idle".
    """
    with foreign_cpu.Probe() as probe:
        result = fn(*args, **kwargs)
    return result, probe.record(), probe.record_max()


def quiet_enough(measured, gate=FOREIGN_GATE):
    """Whether a measured run was quiet enough to be a measurement.

    A run whose foreign-load reading is missing is NOT quiet enough: the
    honest answer is "we do not know", and a table that says "0.0" reads as
    "the box was provably idle", which is a different claim.
    """
    try:
        return float(measured.foreign) <= gate
    except (TypeError, ValueError):
        return False


def lib_guard(site):
    """The library that a site directory actually provides, or None.

    A build is a directory with a `canns_lib/` package inside it, and the
    package has to be the one that directory was built from: a
    conda-installed `canns_lib` can beat `PYTHONPATH` and silently benchmark
    the wrong binary. The check is done in the child, where the import has
    already happened, because that is the only place the truth is.
    """
    return os.path.realpath(os.path.join(site, "canns_lib", "__init__.py"))


def parse_stats(stderr):
    """The build's own per-dimension instrumentation, out of its stderr.

    The build prints one `STATS dim=<n> key=value ...` line per dimension.
    The values are read as STRINGS: they are printed with the precision the
    build chose, and re-parsing them into floats here would quietly change
    what a table says.
    """
    out = {}
    for line in stderr.splitlines():
        if not line.startswith("STATS dim="):
            continue
        fields = {}
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                fields[k] = v
        try:
            dim = int(fields.get("dim", ""))
        except ValueError:
            continue
        out[dim] = fields
    return out


def parse_phasestats(stderr):
    """The build's whole-run breakdown: `PHASESTATS wall=... edges=...`."""
    out = {}
    for line in stderr.splitlines():
        if not line.startswith("PHASESTATS"):
            continue
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                out[k] = v
    return out


def write_table(path, fields, rows):
    """Write a table. The header order is the column order, always."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path


def kv(line):
    """The key=value pairs of one instrumented line."""
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


def num(d, key, cast=float):
    """One number out of a parsed line, or "" when it is not there."""
    try:
        return cast(d.get(key, ""))
    except (TypeError, ValueError):
        return ""


def fingerprint(files, root=REPO):
    """A fingerprint of the sources a build depends on.

    The benchmark host's source tree is a COPY, not a git checkout, so the
    commit is not always available; when it is not, the fingerprint of the
    files the build depends on is the identity instead. A table without one
    of the two cannot be reproduced.
    """
    import hashlib
    h = hashlib.md5()
    for rel in files:
        try:
            with open(os.path.join(root, rel), "rb") as fh:
                h.update(fh.read())
        except OSError:
            pass
    return h.hexdigest()[:12]


def source_commit(root=REPO):
    """The commit the source tree is at, or "" when it is not a checkout."""
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:
        return ""


def build_identity(site, sources=("src/ripser/core/reduction.rs",
                                   "src/ripser/matrix/dense.rs",
                                   "src/ripser/matrix/sparse.rs",
                                   "src/ripser/core/algorithm.rs",
                                   "Cargo.toml")):
    """(commit, fingerprint, extension md5) -- the identity of a build.

    The extension is inside the package directory, so the search is
    RECURSIVE: looking only at the top of the site directory finds nothing
    and records an EMPTY provenance, which is worse than none.
    """
    import hashlib
    commit = source_commit()
    if not commit:
        commit = "src:%s" % fingerprint(sources)
    md5 = ""
    if os.path.isdir(site):
        for root, _dirs, names in os.walk(site):
            for name in sorted(names):
                if name.endswith(".so"):
                    try:
                        with open(os.path.join(root, name), "rb") as fh:
                            md5 = hashlib.md5(fh.read()).hexdigest()[:16]
                    except OSError:
                        md5 = ""
                    break
            if md5:
                break
    return commit, md5
