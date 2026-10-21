#!/usr/bin/env python
"""Gate for the GPU radix sort of the assembled columns.

WHY THIS EXISTS. The barcode md5 that the rest of the harness uses hashes
each dimension's bars AFTER sorting them:

    md5(np.asarray(sorted(map(tuple, x)), dtype="float64").tobytes())

That makes it completely blind to the ORDER of the assembled columns. A radix
sort that produced the right multiset in the wrong order would still pass.
So the barcode md5 is necessary but NOT sufficient for this change, and this
gate adds the check that does constrain it: the device order is compared
element by element against the host order, inside the library, via
`CANNS_RIPSER_VERIFY_SORT=1`.

THREE CHECKS.
  1. SORTVERIFY: for every dimension, the device-sorted-and-deduped column
     sequence equals the host-sorted-and-deduped sequence, element by
     element. `order_mismatch` must be 0 and `len_mismatch` must be False.
  2. BARCODE: the final barcode md5 equals the reference. This is the
     end-to-end check that the whole pipeline still produces the right
     answer, and it also covers the dedup.
  3. KILL SWITCH: with `CANNS_RIPSER_GPU_SORT=0` the host sort runs instead,
     and the barcode md5 must be identical to the GPU-sorted arm. A device
     sort that changed the answer would show up here as a difference.

The two arms are run back to back in one process invocation each, on the
same machine, so a load spike cannot explain a difference.

USAGE
    python sort_gate.py <site_dir> [--expect-md5 A:B:C]

Exit code 0 only if every check passes.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys

import numpy as np

DEFAULT_INPUT = "/tmp/asa1200_capped_only.npy"
DEFAULT_EXPECT = (
    "ee1f5283091dd645a3dd53e6f1a7eac6:"
    "9f0aa8c3ff2716f2fbb9ecd523bb0210:"
    "2d018d8fe9c1895b1b74f837af59ed20"
)

SORTVERIFY_RE = re.compile(
    r"SORTVERIFY dim=(?P<dim>\d+) gpu_n=(?P<gpu_n>\d+) host_n=(?P<host_n>\d+) "
    r"order_mismatch=(?P<order_mismatch>\d+) len_mismatch=(?P<len_mismatch>\w+) "
    r"first=\[(?P<first>[^\]]*)\] s=(?P<s>[\d.]+)"
)
ASSEMBLYSORT_RE = re.compile(
    r"ASSEMBLYSORT dim=(?P<dim>\d+) n=(?P<n>\d+) sorted_on_gpu=(?P<gpu>\w+) "
    r"sort_s=(?P<sort_s>[\d.]+) dedup_s=(?P<dedup_s>[\d.]+)"
)
RADIX_RE = re.compile(
    r"RADIXSORT cols=(?P<cols>\d+) tiles=(?P<tiles>\d+) passes=(?P<passes>\d+) "
    r"s=(?P<s>[\d.]+)"
)
ASSEMBLYGPU_RE = re.compile(r"ASSEMBLYGPU dim=(?P<dim>\d+) cols=(?P<cols>\d+)")


def run_arm(site, matrix, env_extra, timeout):
    """Run one arm. Returns (barcode_md5, stderr_text)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = site
    env.update(env_extra)
    # TWO hashes, and the second one is the important one:
    #
    #   MD5      -- of the bars SORTED. This is the barcode as a set, and it
    #               is what the user cares about.
    #   MD5ORDER -- of the bars in the order the library produced them. The
    #               sorted hash CANNOT see a reordering, and this project
    #               keeps nearly shipping changes that reorder the output
    #               silently (the radix sort changed the assembly order, the
    #               apparent fast path could change the bar order). Any
    #               change that touches how the bars are produced has to
    #               keep this one identical too.
    code = (
        "import hashlib, sys, numpy as np\n"
        "from canns_lib.ripser import ripser\n"
        f"d = np.load({matrix!r})\n"
        "np.fill_diagonal(d, 0.0)\n"
        "cap = float(d[np.isfinite(d)].max()) * 2 + 1.0\n"
        "r = ripser(d, maxdim=2, thresh=cap, coeff=47, distance_matrix=True,\n"
        "           progress_bar=False)['dgms']\n"
        "print('BARS %s MD5 %s MD5ORDER %s' % (\n"
        "    '|'.join(str(len(x)) for x in r),\n"
        "    ':'.join(hashlib.md5(np.asarray(sorted(map(tuple, x)),"
        " dtype='float64').tobytes()).hexdigest() for x in r),\n"
        "    ':'.join(hashlib.md5(np.asarray(list(map(tuple, x)),"
        " dtype='float64').tobytes()).hexdigest() for x in r)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-u", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"arm {env_extra} failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    md5 = None
    md5o = None
    for line in proc.stdout.splitlines():
        if line.startswith("BARS "):
            md5 = line.split("MD5 ", 1)[1].split(" MD5ORDER ")[0].strip()
            md5o = line.split("MD5ORDER ", 1)[1].strip()
    if md5 is None:
        raise SystemExit(f"no BARS line in stdout:\n{proc.stdout}")
    return md5, proc.stderr, md5o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--matrix", default=DEFAULT_INPUT)
    ap.add_argument("--expect-md5", default=DEFAULT_EXPECT)
    ap.add_argument("--expect-md5order", default=None,
                    help="the ORDER-sensitive hash: of the bars as produced")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--csv", default=None,
                    help="write the check results to this CSV")
    args = ap.parse_args()

    failures = []
    notes = []

    # ---- arm 1: device sort, with the in-library differential test ---------
    md5_dev, err_dev, md5o_dev = run_arm(
        args.site,
        args.matrix,
        {
            "CANNS_RIPSER_STATS": "1",
            "CANNS_RIPSER_VERIFY_SORT": "1",
            "CANNS_RIPSER_GPU_SORT": "1",
        },
        args.timeout,
    )
    verifies = [m.groupdict() for m in SORTVERIFY_RE.finditer(err_dev)]
    if not verifies:
        failures.append(
            "no SORTVERIFY line in the GPU-sorted arm: the device sort did not "
            "run, so the order was never checked"
        )
    for v in verifies:
        d = v["dim"]
        if int(v["order_mismatch"]) != 0:
            failures.append(
                f"dim {d}: device order differs from host order in "
                f"{v['order_mismatch']} positions, first={v['first']}"
            )
        if v["len_mismatch"].lower() in ("true", "1"):
            failures.append(
                f"dim {d}: device column count {v['gpu_n']} != host "
                f"{v['host_n']} after dedup"
            )
        notes.append(
            f"dim {d}: gpu_n={v['gpu_n']} host_n={v['host_n']} "
            f"order_mismatch={v['order_mismatch']} s={v['s']}"
        )

    # ---- arm 2: kill switch, host sort ------------------------------------
    md5_host, err_host, md5o_host = run_arm(
        args.site,
        args.matrix,
        {
            "CANNS_RIPSER_STATS": "1",
            "CANNS_RIPSER_GPU_SORT": "0",
        },
        args.timeout,
    )
    if SORTVERIFY_RE.search(err_host):
        failures.append("the host-sorted arm reported SORTVERIFY; the kill switch did not take")

    # ---- the three headline checks -----------------------------------------
    if md5_dev != args.expect_md5:
        failures.append(f"GPU-sorted barcode md5 {md5_dev} != reference {args.expect_md5}")
    if md5_host != args.expect_md5:
        failures.append(f"host-sorted barcode md5 {md5_host} != reference {args.expect_md5}")
    if md5_dev != md5_host:
        failures.append(f"GPU-sorted md5 {md5_dev} != host-sorted md5 {md5_host}")

    # The ORDER-sensitive hash. It is compared between the two arms (the
    # device sort and the host sort must produce the same order) and
    # against the reference when one is supplied.
    if md5o_dev != md5o_host:
        failures.append(
            f"the two arms produced the bars in DIFFERENT orders: "
            f"{md5o_dev} vs {md5o_host}"
        )
    if args.expect_md5order and md5o_dev != args.expect_md5order:
        failures.append(
            f"bar ORDER changed: {md5o_dev} != reference {args.expect_md5order}"
        )

    # ---- timings ------------------------------------------------------------
    radix = [m.groupdict() for m in RADIX_RE.finditer(err_dev)]
    for r in radix:
        notes.append(
            f"radix: cols={r['cols']} tiles={r['tiles']} passes={r['passes']} s={r['s']}"
        )
    sorts = [m.groupdict() for m in ASSEMBLYSORT_RE.finditer(err_dev)]
    for s in sorts:
        notes.append(
            f"assemble dim={s['dim']} n={s['n']} sorted_on_gpu={s['gpu']} "
            f"sort_s={s['sort_s']} dedup_s={s['dedup_s']}"
        )
    gpug = [m.groupdict() for m in ASSEMBLYGPU_RE.finditer(err_dev)]
    for g in gpug:
        notes.append(f"assemblygpu dim={g['dim']} cols={g['cols']}")

    print("=== sort_gate ===")
    for n in notes:
        print("  " + n)
    print(f"  md5 gpu-sort  = {md5_dev}")
    print(f"  md5 host-sort = {md5_host}")
    print(f"  reference     = {args.expect_md5}")
    print(f"  md5order      = {md5o_dev}")

    # The CSV is the data source; the markdown deliverable is rendered from
    # it. Nothing here is hand-transcribed.
    rows = []
    for v in verifies:
        ok = int(v["order_mismatch"]) == 0 and v["len_mismatch"].lower() not in ("true", "1")
        rows.append(dict(
            kind="check",
            check="device column order equals host order",
            detail=("dim %s: %s columns compared element by element, "
                    "%s mismatches"
                    % (v["dim"], v["gpu_n"], v["order_mismatch"])),
            verdict="PASS" if ok else "FAIL",
            caught_by="SORTVERIFY",
            evidence=("SORTVERIFY dim=%s order_mismatch=%s len_mismatch=%s"
                      % (v["dim"], v["order_mismatch"], v["len_mismatch"])),
        ))
    rows.append(dict(
        kind="check",
        check="barcode md5, device-sorted arm",
        detail=md5_dev,
        verdict="PASS" if md5_dev == args.expect_md5 else "FAIL",
        caught_by="barcode md5",
        evidence="sort_gate.py arm 1 (CANNS_RIPSER_GPU_SORT=1)",
    ))
    rows.append(dict(
        kind="check",
        check="barcode md5, host-sorted arm (kill switch)",
        detail=md5_host,
        verdict="PASS" if md5_host == args.expect_md5 else "FAIL",
        caught_by="barcode md5",
        evidence="sort_gate.py arm 2 (CANNS_RIPSER_GPU_SORT=0)",
    ))
    rows.append(dict(
        kind="check",
        check="the two arms produce the bars in the same ORDER",
        detail="%s == %s" % (md5o_dev, md5o_host),
        verdict="PASS" if md5o_dev == md5o_host else "FAIL",
        caught_by="order-sensitive md5",
        evidence="sort_gate.py arms 1 and 2, unsorted bar hash",
    ))
    rows.append(dict(
        kind="check",
        check="device sort and host sort give the same barcode",
        detail="%s == %s" % (md5_dev, md5_host),
        verdict="PASS" if md5_dev == md5_host else "FAIL",
        caught_by="barcode md5",
        evidence="sort_gate.py arms 1 and 2",
    ))
    for r in radix:
        rows.append(dict(
            kind="timing",
            check="device radix sort",
            detail=("%s columns, %s tiles, %s passes, %s s"
                    % (r["cols"], r["tiles"], r["passes"], r["s"])),
            verdict="",
            caught_by="RADIXSORT",
            evidence="RADIXSORT cols=%s tiles=%s passes=%s s=%s"
                     % (r["cols"], r["tiles"], r["passes"], r["s"]),
        ))
    for s in sorts:
        rows.append(dict(
            kind="timing",
            check="host sort / dedup",
            detail=("dim %s: n=%s sorted_on_gpu=%s sort_s=%s dedup_s=%s"
                    % (s["dim"], s["n"], s["gpu"], s["sort_s"], s["dedup_s"])),
            verdict="",
            caught_by="ASSEMBLYSORT",
            evidence="ASSEMBLYSORT dim=%s n=%s sorted_on_gpu=%s sort_s=%s dedup_s=%s"
                     % (s["dim"], s["n"], s["gpu"], s["sort_s"], s["dedup_s"]),
        ))
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(
                f, fieldnames=["kind", "check", "detail", "verdict",
                               "caught_by", "evidence"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote %s" % args.csv)

    if failures:
        print("FAIL")
        for f in failures:
            print("  ! " + f)
        print("--- device-arm stderr (tail 60) ---")
        for line in err_dev.splitlines()[-60:]:
            print("  " + line)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
