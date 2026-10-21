#!/usr/bin/env python3
"""Write the build identity of a table that predates the provenance columns.

A table measured with a given build can state which build that was even if
the harness did not record it per row: the head-to-head table of the same
run records the commit fingerprint, the binary md5, the thread count and the
load average, and the characterisation table was measured with the SAME
build (the same site directory). This writes that identity next to the
table, and the checker verifies that the two tables agree -- so the
sidecar cannot claim a build the rest of the tables do not use.

USAGE
    python write_provenance_sidecar.py --from clean_h2h.csv \
        --out characterize_gridcells.provenance.json
"""
from __future__ import annotations

import argparse
import csv
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    ours = [r for r in rows if r.get("who") == "ours"]
    if not ours:
        raise SystemExit("no 'ours' row in %s" % args.src)
    r = ours[0]
    out = {"commit": r.get("commit", ""), "build_md5": r.get("build_md5", ""),
           "threads": r.get("threads", ""),
           "loadavg_mean": "%.2f" % (sum(float(x["loadavg"]) for x in ours)
                                     / max(1, len(ours))),
           "source_table": args.src}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote %s: %s" % (args.out, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
