#!/bin/bash
# Build the two ablation variants: the shipping build with EXACTLY one
# switch turned off.
#
# Both are pure cargo features, so each variant differs from the shipping
# build by one thing and nothing else -- which is what makes the ablation
# table's rows comparable with each other:
#
#   site_abl_heapoff          `--features ...,binary-heap`
#         the reduction's working heap is `std::collections::BinaryHeap`
#         again, whose `pop` walks the hole to the bottom of the tree
#         instead of stopping early
#
#   site_abl_autosparseoff    `--features ...,legacy-auto-sparse`
#         the sparse representation is chosen whenever the threshold is
#         below the maximum finite distance, without requiring the matrix
#         to actually contain a missing pair
#
# The shipping build itself (site_ship6) is the baseline and is NOT
# rebuilt here -- it must be the same build the rest of the tables used,
# and rebuilding it would make the rows incomparable with them.
#
# FEATURE SET: `gpu` MUST be listed. It is in neither `default` nor
# `[tool.maturin] features` (the released wheel has to build without nvcc),
# and a build without it silently falls back to the CPU apparent pre-pass
# and is ~35% slower -- which reads exactly like a regression in the
# change under test.
set -e
# THE REPOSITORY PATH IS NOT HARDCODED. The driver that calls this script
# (`build_measure_sites.sh`) honours CANNS_BENCH_REPO, and if this script
# built into a different checkout, the driver would report success while the
# ablation sites it then measures came from somewhere else entirely.
REPO=${CANNS_BENCH_REPO:-/home/sichaohe/canns-lib-dev}
cd "$REPO"
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl
export CARGO_TARGET_DIR=$REPO/target

build_variant() {
    name=$1
    extra=$2
    out=/tmp/wh_abl_$name
    echo "=== building $name (features: parallel,hugepage,extension-module,gpu$extra) ==="
    # the output directory is per variant: two wheels in one directory
    # would both be unpacked into the same site, and the second would
    # overwrite the first's extension -- the two variants would then be
    # the same build
    rm -rf "$out"
    maturin build --release --out "$out" \
      --features parallel,hugepage,extension-module,gpu$extra 2>&1 | tail -2
    rm -rf "$REPO/site_abl_$name"
    mkdir -p "$REPO/site_abl_$name"
    cd "$out" && for w in *.whl; do unzip -o -q "$w" -d "$REPO/site_abl_$name"; done
    cd "$REPO"
    # the extension's md5 is the identity of the variant: two variants with
    # the same md5 are the same build, and the ablation would be empty
    so=$(find "$REPO/site_abl_$name" -name "*.so" | head -1)
    echo "BUILT site_abl_$name  md5=$(md5sum "$so" | cut -c1-16)"
}

build_variant heapoff ",binary-heap"
build_variant autosparseoff ",legacy-auto-sparse"
echo "VARIANTS DONE"
