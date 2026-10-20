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
cd ~/canns-lib-dev
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl
export CARGO_TARGET_DIR=$HOME/canns-lib-dev/target

build_variant() {
    name=$1
    extra=$2
    echo "=== building $name (features: parallel,hugepage,extension-module,gpu$extra) ==="
    maturin build --release --out /tmp/wh_abl \
      --features parallel,hugepage,extension-module,gpu$extra 2>&1 | tail -2
    rm -rf ~/canns-lib-dev/site_abl_$name
    mkdir -p ~/canns-lib-dev/site_abl_$name
    cd /tmp/wh_abl && for w in *.whl; do unzip -o -q "$w" -d ~/canns-lib-dev/site_abl_$name; done
    cd ~/canns-lib-dev
    echo "BUILT site_abl_$name"
}

build_variant heapoff ",binary-heap"
build_variant autosparseoff ",legacy-auto-sparse"
echo "VARIANTS DONE"
