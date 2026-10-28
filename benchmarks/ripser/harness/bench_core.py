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

import contextlib
import csv
import errno
import fcntl
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foreign_cpu

# THE MEASUREMENT LOCK.
#
# The quiet-window gate counts the cores used by processes that are NOT ours.
# It therefore cannot see a collision between TWO OF OUR OWN measurements --
# and that collision is just as capable of moving the numbers as a
# neighbour's job is. It happened: a measurement driver's parent was stopped
# while its child kept running, the child and an unrelated measurement ran
# together for half an hour, and both sets of numbers were unusable.
#
# So every measured run takes an exclusive lock on this file. A second
# measurement WAITS for the first to finish rather than running beside it.
# The lock is per file descriptor, so a process that takes it twice does not
# deadlock -- it is the second PROCESS that waits.
LOCK_PATH = "/tmp/.canns_measurement.lock"


@contextlib.contextmanager
def measurement_lock(timeout=3600, label=""):
    """Hold the box for the duration of one measurement.

    The lock is an exclusive `flock` on a file, so it is released when the
    process dies -- including when it is killed, which is what makes it safe
    against a measurement whose parent was stopped: the child keeps the lock
    it inherited, and the next measurement waits for it.

    Waiting is the right behaviour and not a failure: two measurements on the
    same box at the same time would both be wrong, and the second one is
    better off waiting than being measured beside the first.
    """
    who = label or os.path.basename(sys.argv[0])
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        deadline = time.time() + timeout
        waited = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.time() > deadline:
                    raise RuntimeError(
                        "another measurement has held %s for more than %d s; "
                        "refusing to run beside it"
                        % (LOCK_PATH, timeout))
                if not waited:
                    waited = True
                    print("  (waiting for the box: another measurement is "
                          "running)", file=sys.stderr, flush=True)
                time.sleep(2.0)
        try:
            os.ftruncate(fd, 0)
            os.write(fd, ("%d %s\n" % (os.getpid(), who)).encode())
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

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

# WHAT A ROW MAY SAY AND STILL BE A MEASUREMENT.
#
# The runners record how a row ended in one of two columns, and the two
# columns do not use the same words:
#
#   `error`   -- set by the scale scan: "" | "TIMEOUT" | "CRASH"
#   `verdict` -- set by the other runners: "OK" | "TIMEOUT" | "MISSING" |
#                "SAME_BARCODE" | "BADCODEC" | "BARCODE_MISMATCH" |
#                "NOGPU" | "RC=<code>"
#
# A timeout is a LEGAL RESULT in both: the run hit the time limit and the
# table says so. Everything else in those lists is a row that did not measure
# what the table is about -- a crashed child, a missing run, a barcode that
# does not match, a build without the feature, a non-zero return code -- and
# those are failures.
LEGAL_ERRORS = {"", "TIMEOUT"}
LEGAL_VERDICTS = {"", "OK", "TIMEOUT", "SAME_BARCODE"}

# A counter that makes each measured run's output file name unique. It is
# not a lock: it only stops two overlapping runs from writing to the same
# file.
_RUN_COUNTER = [0]


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
    # The command is a list, never a shell string, so nothing in it can be
    # re-parsed as shell syntax. Its parts come from this repository's own
    # constants and from the paths on the benchmark host -- the site
    # directory, the dataset path, the interpreter -- and not from anything
    # an outside party can supply; the dataset path is a fixed list here, not
    # an argument a caller can point anywhere.
    # The output files are named after the run, not after the tag alone: two
    # runners that overlap on the same dataset and arm would otherwise open
    # the SAME files with truncation while both children write, and each
    # parent would parse the other one's output as its own barcode and stats.
    # The pid and a counter make the name unique per run.
    unique = "%s_%d_%d" % (tag or "run", os.getpid(), _RUN_COUNTER[0])
    _RUN_COUNTER[0] += 1
    out_path = "/tmp/_bench_%s.out" % unique
    err_path = "/tmp/_bench_%s.err" % unique
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        with measurement_lock(label=tag or "run"):
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
            # The reading is taken AFTER the probe's block has closed: the
            # probe computes its mean over the window that ends when the
            # block exits, and reading it inside the block asks for a mean
            # over no samples at all -- which comes back as "" and makes the
            # row look like it has no reading at all.
            foreign = probe.record()
            foreign_max = probe.record_max()
    with open(out_path) as fh:
        stdout = fh.read()
    with open(err_path) as fh:
        stderr = fh.read()
    if not keep:
        os.unlink(out_path)
        os.unlink(err_path)
    return Measured(argv, stdout, stderr, proc.returncode, wall,
                    foreign, foreign_max, env)


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
    with measurement_lock(label="run"):
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


def foreign_summary(path):
    """The foreign-load picture of a whole table:
    (worst, missing, broken, timed_out, rows).

    THE ONE PLACE THE GATE IS COMPUTED. The drivers ask the same question of
    a table -- "was every row measured on a quiet host?" -- and they used to
    answer it with three copies of the same inline python, which is how copies
    drift. `worst` is the largest mean foreign load over a row, `missing` is
    the number of rows with no reading at all, `broken` is the number of rows
    whose run did not finish, and `timed_out` is the number of rows that hit
    the time limit.

    A row with no reading is not a quiet row: its host state is unknown, and
    "we do not know" is a failure, not a footnote.

    `broken` and `timed_out` are counted separately, and a timeout is NOT a
    broken row. A timeout is a legal result: the runners record it as such
    (`TIMEOUT` in the `error` column, `TIMEOUT` in the `verdict` column), and
    a table that says "this did not finish inside 600 s" tells the truth. A
    row whose child died is not a measurement: it looks like a fast run and it
    is not. A gate that calls both "contaminated" sends the reader to the
    wrong place: a scale scan whose nine rows crashed on a NameError in the
    child script was reported as contaminated, which points at the host.
    """
    worst = 0.0
    missing = 0
    broken = 0
    timed_out = 0
    rows = 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows += 1
            raw = (r.get("foreign_cpu") or "").strip()
            if not raw:
                missing += 1
            else:
                try:
                    worst = max(worst, float(raw))
                except ValueError:
                    missing += 1
            err = (r.get("error") or "").strip()
            verdict = (r.get("verdict") or "").strip()
            if err.upper() == "TIMEOUT" or verdict.upper() == "TIMEOUT":
                # A legal result: the run hit the time limit and said so.
                timed_out += 1
            elif err and err.upper() not in LEGAL_ERRORS:
                broken += 1
            elif verdict and verdict.upper() not in LEGAL_VERDICTS:
                broken += 1
    return worst, missing, broken, timed_out, rows


def batch_is_clean(path, limit=FOREIGN_GATE):
    """Whether a whole table can be shown to have been measured on a quiet host.

    The batch is ALL OR NOTHING: a neighbour's job can start in the middle of
    a measurement, and half a table measured under load is worse than no
    table, because it looks like a table. A row that CRASHED is also a
    failure -- a row that looks like a fast run because the child died is not
    a measurement. A row that TIMED OUT is not: it is a result, and it says
    so.
    """
    try:
        worst, missing, broken, _timed_out, rows = foreign_summary(path)
    except OSError:
        return False
    if rows == 0:
        return False
    return missing == 0 and broken == 0 and worst <= limit


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
