#!/bin/bash
# Rebuild every build the measurement tables are produced from.
#
# WHY THIS IS A DRIVER AND NOT A ONE-OFF. The tables in this repository were
# produced by specific builds, and a table is only comparable with another
# table when the two came from the same code. When the engine changes, every
# build that a table was measured with has to be rebuilt, or the new table
# silently compares the new code against the old one.
#
# The builds:
#
#   site_ship6        the shipping build, GPU features. The baseline of the
#                     ablation table and the "gpu" arm of the standard suite.
#   site_cpu_final    the same code WITHOUT the gpu feature. The "cpu" arm of
#                     the standard suite and the o3 profile.
#   site_abl_*        the ablation variants, one cargo feature each, built by
#                     build_abl_variants.sh.
#
# THE OLD BUILDS ARE KEPT, RENAMED. They are the reference the new tables are
# compared against -- the barcode cross-check needs a build of the OLD code to
# compare against -- so they are moved aside with a suffix, not deleted.
#
# FEATURE SET: `gpu` MUST be listed for the shipping build. It is in neither
# `default` nor `[tool.maturin] features` (the released wheel has to build
# without nvcc), and a build without it silently falls back to the CPU
# apparent pre-pass and is ~35% slower -- which reads exactly like a
# regression in the change under test.
# NOT `set -u`. Conda's activate script reads ADDR2LINE, which is not set in
# a non-interactive shell, and `set -u` turns that into a fatal error that
# kills this script before it builds anything. The same trap is in
# corpus_quiet_run.sh.
set -eo pipefail

REPO=${CANNS_BENCH_REPO:-/home/sichaohe/canns-lib-dev}
STAMP=${CANNS_BENCH_STAMP:-pre_early_exit}
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl
export CARGO_TARGET_DIR=$REPO/target

cd "$REPO"

# THE LOCK, shared with sync_engine_to_server.sh. The sync replaces the
# engine files one at a time, so a build that starts in the middle of it
# would compile a MIXTURE of two revisions -- every file passes its checksum
# and the result does not describe any single revision. So the two cannot
# overlap: the build waits for the sync to finish, and holds the lock while
# it runs.
LOCK=/tmp/.canns_engine_sync.lock
waited=0
while [ -e "$LOCK" ]; do
    if [ "$waited" -ge 120 ]; then
        echo "REFUSING: the sync lock $LOCK has been held for more than two minutes"
        exit 1
    fi
    echo "  waiting for the engine sync to finish ($LOCK)..."
    sleep 5
    waited=$((waited + 5))
done
touch "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# The old builds are moved aside, not deleted: the barcode cross-check needs
# a build of the OLD code to compare the new code against.
for s in site_ship6 site_cpu_final; do
    if [ -d "$REPO/$s" ] && [ ! -d "$REPO/${s}_$STAMP" ]; then
        mv "$REPO/$s" "$REPO/${s}_$STAMP"
        echo "moved aside: $s -> ${s}_$STAMP"
    fi
done

build_site() {
    name=$1
    features=$2
    out=/tmp/wh_$name
    echo "=== building $name (features: $features) ==="
    rm -rf "$out" "$REPO/site_$name"
    maturin build --release --out "$out" --features "$features" 2>&1 | tail -2
    mkdir -p "$REPO/site_$name"
    (cd "$out" && for w in *.whl; do unzip -o -q "$w" -d "$REPO/site_$name"; done)
    so=$(find "$REPO/site_$name" -name "*.so" | head -1)
    echo "BUILT site_$name  md5=$(md5sum "$so" | cut -c1-16)"
}

build_site ship6     parallel,hugepage,extension-module,gpu
build_site cpu_final parallel,hugepage,extension-module

# The ablation variants, each one cargo feature away from the shipping build.
# The repository path is EXPORTED, so the variant builder builds into the
# same checkout as the builds above: if it built somewhere else, this script
# would report success while the ablation sites it then measures came from
# another tree entirely.
export CANNS_BENCH_REPO="$REPO"
bash "$REPO/benchmarks/ripser/drivers/build_abl_variants.sh"

echo "ALL SITES BUILT"
for s in site_ship6 site_cpu_final site_abl_heapoff site_abl_autosparseoff; do
    so=$(find "$REPO/$s" -name "*.so" 2>/dev/null | head -1)
    [ -n "$so" ] && echo "  $s  $(md5sum "$so" | cut -c1-16)"
done
