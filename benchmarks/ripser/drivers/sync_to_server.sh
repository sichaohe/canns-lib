#!/bin/bash
# Copy the benchmark scripts to the benchmark host, atomically, and CHECK.
#
# THE FAILURE THIS GUARDS AGAINST. `cat > dst` truncates the target the
# moment the command starts, so a sync that is interrupted -- or one that
# is chained with a background launch and dies with the ssh session --
# leaves a ZERO-BYTE file behind. The remote then runs a stale or empty
# script, and the error points somewhere else entirely: we lost an hour
# to "module has no attribute 'provenance'" that was really an empty file.
#
# So: write to a temporary file, move it into place, and then assert that
# every file landed with the expected size and md5. A zero-length file is
# refused on both ends, and the empty file's md5 is
# d41d8cd98f00b204e9800998ecf8427e.
#
# USAGE
#     bash benchmarks/ripser/drivers/sync_to_server.sh
set -euo pipefail

HOST=${CANNS_BENCH_HOST:-222.29.101.57}
PORT=${CANNS_BENCH_PORT:-1693}
KEY=${CANNS_BENCH_KEY:-$HOME/.ssh/sichaohe_key}
# this script lives in `drivers/`, so the tree it syncs is its parent
SRC=$(cd "$(dirname "$0")/.." && pwd)
DST=${CANNS_BENCH_DIR:-/home/sichaohe/canns-lib-dev/benchmarks/ripser}

# The paths are relative to the tree, so a file in `checkers/` is listed as
# `checkers/check_x.py` and lands in the same subdirectory on the host.
FILES="harness/scale_scan.py harness/standard_bench.py harness/unified_ablation.py \
harness/t_arm_equivalence.py harness/foreign_cpu.py harness/o3_profile.py \
harness/o3_cpu_scaling.py harness/clean_h2h.py harness/h2h_coeff_ab.py \
drivers/quiet_window_watch.sh drivers/corpus_quiet_run.sh \
tools/cross_batch.py \
checkers/check_scale_scan.py checkers/check_standard_bench.py \
checkers/check_unified_ablation.py checkers/check_gridcells_bench.py \
checkers/check_clean_h2h.py checkers/check_nsfix_ab.py \
checkers/check_bounds_ab.py checkers/check_validate_gridcells.py \
checkers/check_characterize_gridcells.py checkers/check_o3_profile.py \
checkers/check_o3_cpu_scaling.py checkers/check_h2h_coeff_ab.py \
checkers/check_reduce_split.py"

SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15 sichaohe@$HOST"

echo "syncing to $HOST:$PORT$DST"
for f in $FILES; do
    if [ ! -f "$SRC/$f" ]; then
        echo "  MISSING LOCALLY: $f" >&2
        exit 1
    fi
    local_size=$(stat -f%z "$SRC/$f" 2>/dev/null || stat -c%s "$SRC/$f")
    if [ "$local_size" -eq 0 ]; then
        echo "  REFUSING: $f is empty locally" >&2
        exit 1
    fi
    local_md5=$(md5 -q "$SRC/$f")
    # the write and the move are one command, and the move is atomic: the
    # target is never observed half-written. The subdirectory has to exist
    # on the host first, or the move fails and the file is left in /tmp.
    $SSH "mkdir -p $(dirname "$DST/$f")"
    cat "$SRC/$f" | $SSH "cat > /tmp/_sync_one && mv /tmp/_sync_one $DST/$f"
    # The md5 and the size come back as two LINES, and `read` stops at the
    # first newline -- so the size was silently empty, the "landed empty"
    # assertion below never fired, and the md5 comparison always failed.
    # Join the lines with a space first, then read them.
    remote_info=$($SSH "md5sum $DST/$f | cut -d' ' -f1; stat -c%s $DST/$f" | tr '\n' ' ')
    read -r remote_md5 remote_size <<<"$remote_info"
    if [ -z "$remote_size" ]; then
        echo "  REFUSING: $f -- the size of the copy on the host could not be "
        echo "            read, so its integrity cannot be asserted" >&2
        exit 1
    fi
    if [ "$remote_size" -eq 0 ]; then
        echo "  REFUSING: $f landed EMPTY on the host" >&2
        exit 1
    fi
    if [ "$remote_md5" != "$local_md5" ] || [ "$remote_size" != "$local_size" ]; then
        echo "  MISMATCH: $f local=$local_md5/$local_size remote=$remote_md5/$remote_size" >&2
        exit 1
    fi
    printf "  ok  %-34s %8s bytes  %s\n" "$f" "$remote_size" "$remote_md5"
done
echo "all $(( $(echo $FILES | wc -w) )) files synced and verified"
