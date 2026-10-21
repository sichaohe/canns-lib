#!/usr/bin/env python
"""Negative self-test for sort_gate.py.

A gate that only ever prints PASS proves nothing. This script builds three
DELIBERATELY BROKEN variants of the library and checks that sort_gate.py
rejects each one:

  A. bad_key    -- the device sort key orders the tie-break index DESCENDING
                    instead of ascending. The multiset of columns is
                    unchanged, only their order is wrong.
  B. bad_passes -- the radix sort runs 12 passes instead of 16, so the top
                    16 bits of the diameter key are never used. The sort is
                    incomplete.
  C. bad_swap   -- the device-to-host repack swaps the diameter and the
                    index. The ORDER is still self-consistent, so the
                    in-library order check passes; only the end-to-end
                    barcode check can catch this one.

Each variant is built on the server, installed into its own site directory,
and run through sort_gate.py. A variant that the gate PASSes is a FAILURE of
this script, not a success of the gate.

⚠️ THE VARIANTS MUST BE ISOLATED. Before each variant this script restores
EVERY file it snapshots, not just the one it is about to patch. An earlier
version restored only the current file, so the previous variant's breakage
was still in the tree: variants 2 and 3 were variant 1 PLUS the new change,
and they produced byte-identical gate output -- which is exactly what made
the bug visible. The script also asserts that the rebuilt .so differs from
the baseline .so, so a stale build cannot masquerade as a test.

USAGE
    python neg_test.py <repo_root_on_server> [--gate-timeout 1500]
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

VARIANTS = [
    {
        "name": "bad_key",
        "file": "kernel/ripser_kernels.cu",
        "old": "return ((u64)(~s) << 32) | (unsigned int)((unsigned long long)c.i);",
        "new": "return ((u64)(~s) << 32) | (unsigned int)(~(unsigned long long)c.i);",
        "why": "device key orders the tie-break index descending",
    },
    {
        "name": "bad_passes",
        "file": "src/ripser/gpu_assembly.rs",
        "old": "const PASSES: usize = 16; // 64-bit key / 4 bits per pass",
        "new": "const PASSES: usize = 12; // NEGATIVE TEST: 12 of 16 passes",
        "why": "radix sort runs 12 of 16 passes, leaving the high key bits unsorted",
    },
    {
        "name": "bad_swap",
        "file": "src/ripser/gpu_assembly.rs",
        "old": "                out.par_extend((0..n).into_par_iter().map(|i| {\n"
               "                    let d = f32::from_bits(raw[i * 2] as u32);\n"
               "                    let idx = raw[i * 2 + 1];",
        "new": "                out.par_extend((0..n).into_par_iter().map(|i| {\n"
               "                    let d = f32::from_bits(raw[i * 2 + 1] as u32);\n"
               "                    let idx = raw[i * 2];",
        "why": "the D2H repack swaps the diameter and the index",
    },
]

PY = "/home/sichaohe/miniconda3/envs/rl/bin/python"


def sh(cmd, cwd=None, timeout=None, check=True, env=None):
    p = subprocess.run(
        cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True,
        text=True, timeout=timeout, env=env,
    )
    if check and p.returncode != 0:
        raise SystemExit(
            f"command failed ({p.returncode}): {cmd}\n"
            f"--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}"
        )
    return p


def patch(path, old, new):
    """Replace `old` with `new`, exactly once. Returns the original text."""
    with open(path, "r") as f:
        src = f.read()
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"patch target not unique in {path}: found {n} occurrences")
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--gate-timeout", type=int, default=1500)
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)

    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":/usr/local/cuda/bin:" + env["PATH"]
    env["PYO3_PYTHON"] = PY

    gate = os.path.join(repo, "benchmarks/ripser/sort_gate.py")
    base_site = os.path.join(repo, "site_radix")
    if not os.path.isdir(base_site):
        raise SystemExit(f"missing baseline site {base_site}")

    # Snapshot the files we mutate so they can be restored byte for byte.
    snapshot = {}
    for v in VARIANTS:
        p = os.path.join(repo, v["file"])
        with open(p, "rb") as f:
            snapshot[p] = f.read()

    results = []
    try:
        for v in VARIANTS:
            name = v["name"]
            print(f"\n=== variant {name}: {v['why']} ===", flush=True)
            path = os.path.join(repo, v["file"])
            # ⚠️ Restore EVERY snapshotted file, not just this variant's. An
            # earlier version restored only the file it was about to patch,
            # so the PREVIOUS variant's breakage was still in the tree: the
            # second and third variants were the first variant PLUS the new
            # one, and they produced byte-identical gate output. The variants
            # have to be independent, or the negative test proves nothing.
            for p, data in snapshot.items():
                with open(p, "wb") as f:
                    f.write(data)
            patch(path, v["old"], v["new"])

            t0 = time.time()
            build = sh(
                ["cargo", "build", "--release", "--features",
                 "extension-module,gpu,hugepage"],
                cwd=repo, env=env, timeout=1800,
            )
            print(f"  build: {build.stderr.strip().splitlines()[-1] if build.stderr.strip() else 'ok'}",
                  flush=True)

            so = os.path.join(repo, "target/release/libcanns_lib.so")
            so_md5 = hashlib.md5(open(so, "rb").read()).hexdigest()
            base_md5 = hashlib.md5(
                open(os.path.join(base_site, "canns_lib/canns_lib.so"), "rb").read()
            ).hexdigest()
            if so_md5 == base_md5:
                raise SystemExit(
                    f"variant {name}: the rebuilt .so is byte-identical to the "
                    "baseline -- the build did not pick up the change, so the "
                    "test would be measuring nothing"
                )
            print(f"  .so md5 {so_md5} (baseline {base_md5})", flush=True)

            site = os.path.join(repo, f"site_neg_{name}")
            if os.path.isdir(site):
                shutil.rmtree(site)
            shutil.copytree(base_site, site)
            shutil.copy2(so, os.path.join(site, "canns_lib/canns_lib.so"))

            genv = dict(env)
            genv["PYTHONPATH"] = site
            gatep = sh(
                [PY, "-u", gate, site],
                cwd=os.path.expanduser("~"), env=genv, timeout=args.gate_timeout,
                check=False,
            )
            out = gatep.stdout + gatep.stderr
            passed = "\nPASS" in out and "FAIL" not in out
            verdict = "GATE-PASSED(BAD!)" if passed else "GATE-REJECTED(good)"
            print(f"  {verdict}  [{time.time() - t0:.0f}s]", flush=True)
            for line in out.splitlines():
                if line.strip().startswith("!") or line.strip() == "FAIL" or \
                        "order_mismatch=" in line or "len_mismatch=" in line or \
                        "md5" in line:
                    print("    " + line.strip(), flush=True)
            caught = []
            if not passed:
                if "order_mismatch=" in out and "order_mismatch=0 " not in out:
                    caught.append("SORTVERIFY(order)")
                if "len_mismatch=true" in out:
                    caught.append("SORTVERIFY(length)")
                if "md5" in out and "reference" in out:
                    caught.append("barcode md5")
            results.append((name, passed, out, v["why"], ", ".join(caught) or "none"))
    finally:
        for p, data in snapshot.items():
            with open(p, "wb") as f:
                f.write(data)
        print("\nrestored mutated files", flush=True)

    print("\n=== negative self-test summary ===")
    bad = 0
    csv_path = os.path.join(repo, "benchmarks/ripser/sort_neg_results.csv")
    with open(csv_path, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["kind", "check", "detail", "verdict", "caught_by", "evidence"])
        for name, passed, _, why, caught in results:
            if passed:
                bad += 1
                print(f"  {name}: GATE PASSED a deliberately broken build -- the gate is BLIND")
            else:
                print(f"  {name}: rejected, as it must be")
            w.writerow([
                "negative",
                name,
                why,
                "GATE-PASSED(BAD!)" if passed else "GATE-REJECTED(good)",
                caught,
                "sort_neg_test.py: %s" % name,
            ])
    print("wrote %s" % csv_path)
    if bad:
        print("NEGATIVE SELF-TEST FAILED: the gate accepted a broken build")
        return 1
    print("NEGATIVE SELF-TEST PASSED: every broken build was rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
