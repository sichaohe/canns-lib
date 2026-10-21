#!/usr/bin/env python3
"""The two figures of the load-characterisation section.

Figure 1 -- the distributions: the missing-pair fraction and the
apparent-pair fraction at dim 1 and dim 2, over the whole corpus. This is
what replaces a single instance's number: the claim is "the property
holds across the corpus", and the figure is the evidence.

Figure 2 -- the phase decomposition of the wall time, one stacked bar
per matrix, sorted by the size of the dim-2 reduction. This is what
replaces the assertion "this is not a reduction problem": the reader can
see how much of the time is in the reduction loop, how much in the
apparent pre-pass, how much in the pivot table build, and how much
elsewhere -- and judge the claim themselves.

USAGE
    python plot_characterization.py characterize_gridcells.csv --outdir figures
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def col(rows, key, f=float):
    out = []
    for r in rows:
        v = (r.get(key) or "").strip()
        if not v:
            continue
        try:
            out.append(f(v))
        except ValueError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="characterize_gridcells.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    os.makedirs(args.outdir, exist_ok=True)
    n = len(rows)
    print("matrices: %d" % n)

    # ---------------- figure 1: the distributions -------------------------
    inf = col(rows, "inf_frac")
    a1 = col(rows, "dim1_apparent_frac")
    a2 = col(rows, "dim2_apparent_frac")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, v, title, xlabel in (
            (axes[0], inf, "missing point pairs", "fraction of pairs that are +inf"),
            (axes[1], a1, "apparent pairs, dim 1", "fraction of the columns"),
            (axes[2], a2, "apparent pairs, dim 2", "fraction of the columns")):
        if v:
            ax.hist(v, bins=24, color="#4C72B0", edgecolor="white")
            ax.set_title("%s\nmedian %.4f  (min %.4f, max %.4f)"
                         % (title, sorted(v)[len(v) // 2], min(v), max(v)),
                         fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("matrices", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("The load the grid-cell recordings produce (n=%d matrices)" % n,
                 fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(args.outdir, "characterization_distributions.png")
    fig.savefig(p1, dpi=200)
    print("wrote %s" % p1)

    # ---------------- figure 2: the phase decomposition --------------------
    parts = {"dim-2 reduction loop": [], "dim-1 reduction loop": [],
             "GPU apparent pre-pass": [], "sharded pivot build": [],
             "assembly": [], "other": []}
    for r in rows:
        try:
            wall = float(r.get("ph_wall_s") or r["wall_s"])
        except (TypeError, ValueError):
            continue
        if wall <= 0:
            continue
        def g(k):
            try:
                return float(r.get(k) or 0.0)
            except ValueError:
                return 0.0
        red = g("ph_reduce_s")
        asm = g("ph_assemble_s")
        # SHARES of the library wall, not seconds: the section quotes the
        # composition, and the corpus spans two orders of magnitude in
        # instance cost.
        parts["dim-2 reduction loop"].append(g("dim2_loop_s") / wall)
        parts["dim-1 reduction loop"].append(g("dim1_loop_s") / wall)
        parts["GPU apparent pre-pass"].append(g("dim2_gpu_pre_s") / wall)
        parts["sharded pivot build"].append(g("dim2_pivot_build_s") / wall)
        parts["assembly"].append(asm / wall)
        parts["other"].append(max(0.0, wall - red - asm) / wall)
    if parts["dim-2 reduction loop"]:
        m = len(parts["dim-2 reduction loop"])
        order = sorted(range(m), key=lambda i: parts["dim-2 reduction loop"][i])
        colors = {"dim-2 reduction loop": "#C44E52", "dim-1 reduction loop": "#DD8452",
                  "GPU apparent pre-pass": "#4C72B0", "sharded pivot build": "#55A868",
                  "assembly": "#8172B3", "other": "#CCCCCC"}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.4))

        # (a) the SHARE of the wall time per phase, across the corpus: a
        #     box per phase is what the section quotes, and 97 stacked bars
        #     would be unreadable anyway.
        names = list(parts)
        data = [[100.0 * v for v in parts[n]] for n in names]
        bp = ax1.boxplot(data, labels=names, patch_artist=True, showfliers=True,
                          flierprops=dict(marker=".", markersize=3,
                                          markerfacecolor="#888888"))
        for patch, name in zip(bp["boxes"], names):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.85)
        ax1.set_ylabel("share of the library wall time (%)", fontsize=8)
        ax1.set_title("Where the time goes, across the corpus", fontsize=10)
        ax1.tick_params(axis="x", labelsize=7, rotation=25)
        ax1.tick_params(axis="y", labelsize=7)

        # (b) the absolute library wall, on a log axis so the one heavy
        #     instance is visible without hiding the other 96.
        walls = [float(r.get("ph_wall_s") or r.get("wall_s") or 0)
                 for r in rows]
        walls = [w for w in walls if w > 0]
        if walls:
            ax2.hist(walls, bins=24, color="#4C72B0", edgecolor="white")
            ax2.set_yscale("log")
            ax2.set_xlabel("library wall (s)", fontsize=8)
            ax2.set_ylabel("matrices (log)", fontsize=8)
            ax2.set_title("The wall time per matrix (one heavy instance:\n"
                          "%.0f s against a median of %.1f s)"
                          % (max(walls), sorted(walls)[len(walls) // 2]),
                          fontsize=9)
            ax2.tick_params(labelsize=7)
        fig.suptitle("The load the grid-cell recordings produce (maxdim=2)",
                     fontsize=10)
        fig.tight_layout()
        p2 = os.path.join(args.outdir, "characterization_phases.png")
        fig.savefig(p2, dpi=200)
        print("wrote %s" % p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
