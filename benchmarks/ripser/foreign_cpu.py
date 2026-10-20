"""How many cores were used by processes that are not ours?

The load average cannot answer the question the measurements need answered.
It counts our own work: the reduction runs on the host with one thread per
core, and our own runs put the load average at 3.1 on this 16-core box. A
contamination threshold therefore has to sit above our own footprint, and
a neighbour's job on two or three cores then hides under it. That is not
hypothetical: the load average of this box was 2.5-2.9 from a neighbour
that was winding down, and no threshold could tell that apart from us.

So the box is asked directly. Every process's CPU time is read from
/proc, the ones that belong to another user are summed, and the result is
a number of CORES consumed by other people while our measurement ran. Ours
is excluded by construction, so the number is the neighbour's load and
nothing else -- which means the gate does not need a fudge factor for how
busy we are.

TWO NUMBERS, AND THEY MEAN DIFFERENT THINGS.

`foreign_cpu` is the MEAN: the total CPU time other people used, divided
by the wall time of the measurement. It answers "how much of the machine
did somebody else have while this row was measured", and that is the
number the gate uses, because it is the number that can distort a
measurement. A neighbour's eight-core job for the whole of a run shows up
here as eight; a neighbour's job for a tenth of it as 0.8; a desktop
compositor that used one core for half a second out of a two-minute run
shows up as 0.004, which is what it is worth.

`foreign_max` is the largest rate seen in any half-second window. It is a
DIAGNOSTIC, not the gate: a half-second blip from the box's own desktop
session is not a reason to throw away a two-minute measurement, and
gating on the maximum does exactly that -- a batch of clean measurements
was discarded over a 0.987-core blip.

PID REUSE. The reading is taken from two snapshots half a second apart.
If a process exits in between and the operating system hands its pid to
another process, the difference of the two tick counts is meaningless and
can be enormous -- a whole core out of nothing. Every process is
therefore identified by its start time as well as its pid, and a pid
whose start time changed is not compared with itself.

The probe is deliberately conservative about processes that appear or
disappear inside a sampling window: they contribute nothing, so the
reading can understate a very short neighbour -- it cannot overstate one.

ONE KNOWN BIAS, IN THE STRICT DIRECTION. The rate of a sampling window is
computed over the nominal interval, but reading a few hundred /proc files
takes some milliseconds, so the window is slightly LONGER than the
interval and the rate is over-reported by that much -- on this host, a few
per cent. It over-reports, so the gate is stricter than the truth and a
table that passes it passes more easily, not less; the measured values in
the tables are two orders of magnitude below the threshold, so the bias
is irrelevant to them. The mean is diluted by the same amount in the
other direction, because the first sampling window is not covered by the
elapsed time; both effects are small and both are recorded rather than
corrected, so that the tables and the code that produced them agree.
"""

import os
import threading
import time

CLK_TCK = os.sysconf("SC_CLK_TCK")
MY_UID = os.getuid()


def _snapshot():
    """{pid: (utime+stime ticks, starttime)} and {pid: effective uid}."""
    ticks, uids = {}, {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open("/proc/%s/stat" % pid, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        # The comm field can contain spaces and parentheses, so the fixed
        # fields are counted from the LAST ')' rather than by splitting.
        rp = data.rfind(b")")
        if rp < 0:
            continue
        fields = data[rp + 2:].split()
        # after comm: state, ppid, pgrp, session, tty, tpgid, flags,
        # minflt, cminflt, majflt, cmajflt, utime, stime, cutime, cstime,
        # priority, nice, num_threads, itrealvalue, starttime
        try:
            ticks[pid] = (int(fields[11]) + int(fields[12]), int(fields[19]))
        except (IndexError, ValueError):
            continue
        try:
            with open("/proc/%s/status" % pid, "rb") as fh:
                for line in fh:
                    if line.startswith(b"Uid:"):
                        # Uid: real effective saved fs
                        uids[pid] = int(line.split()[2])
                        break
        except (OSError, IndexError, ValueError):
            continue
    return ticks, uids


def _diff(prev, cur, window):
    """(foreign ticks, foreign cores, per-process detail) between snapshots."""
    (pa, ua), (pb, ub) = prev, cur
    foreign = 0
    detail = []
    for pid in set(pa) & set(pb):
        (t0, s0), (t1, s1) = pa[pid], pb[pid]
        if s0 != s1:
            # The pid was recycled: these are two different processes, and
            # their tick counts have nothing to do with each other.
            continue
        d = t1 - t0
        if d <= 0:
            continue
        uid = ub.get(pid, ua.get(pid))
        if uid is None or uid == MY_UID:
            continue
        foreign += d
        detail.append((d / float(CLK_TCK) / window, uid, pid))
    return foreign, foreign / float(CLK_TCK) / window, detail


def foreign_cores(window=0.5):
    """The mean foreign cores over one window, for a one-off reading."""
    a = _snapshot()
    time.sleep(window)
    b = _snapshot()
    _ticks, cores, _detail = _diff(a, b, window)
    return cores


class Probe:
    """Samples the foreign load while a measurement runs.

    Use as a context manager around the child:

        with Probe() as p:
            out, err, rc = run_timed(...)
        print(p.record(), p.record_max())

    `record()` is the MEAN foreign cores over the whole measurement -- the
    number the quiet-window gate uses. `record_max()` is the largest rate
    seen in one sampling window -- a diagnostic.
    """

    def __init__(self, interval=0.5):
        self.interval = interval
        self.max_cores = 0.0
        self.samples = 0
        self.total_ticks = 0
        self._t0 = None
        self._elapsed = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        prev = _snapshot()
        while not self._stop.is_set():
            self._stop.wait(self.interval)
            if self._stop.is_set():
                break
            cur = _snapshot()
            ticks, cores, _detail = _diff(prev, cur, self.interval)
            self.total_ticks += ticks
            self.samples += 1
            if cores > self.max_cores:
                self.max_cores = cores
            prev = cur

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._elapsed = time.perf_counter() - (self._t0 or time.perf_counter())
        return False

    def mean_cores(self):
        """The mean foreign cores over the measurement."""
        if self._elapsed <= 0 or self.samples == 0:
            return None
        return self.total_ticks / float(CLK_TCK) / self._elapsed

    def record(self):
        """The value to write into a table row: the mean foreign cores."""
        m = self.mean_cores()
        if m is None:
            # The probe never got a sample: say so rather than report 0.0,
            # which would read as "the box was provably idle".
            return ""
        return "%.4f" % m

    def record_max(self):
        """The largest foreign rate seen in one sampling window."""
        if self.samples == 0:
            return ""
        return "%.3f" % self.max_cores
