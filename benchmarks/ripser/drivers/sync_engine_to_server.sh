#!/bin/bash
# Copy the ENGINE sources to the benchmark host, atomically, and CHECK.
#
# WHY THIS IS SEPARATE FROM sync_to_server.sh. That one syncs the benchmark
# tree (harness/checkers/tools). This one syncs the RUST sources, and it has
# to be a different list because the two trees have different failure modes:
# the benchmark scripts are read by a python process that reports its own
# errors, while a stale .rs file compiles into a binary that runs and produces
# numbers. A benchmark script that did not land fails loudly; an engine file
# that did not land fails silently, as a measurement of the wrong build.
#
# So: write to a temporary file, move it into place, and assert that every
# file landed with the expected size and md5. A zero-length file is refused
# on both ends, and the empty file's md5 is
# d41d8cd98f00b204e9800998ecf8427e.
#
# USAGE
#     bash benchmarks/ripser/drivers/sync_engine_to_server.sh
#
# The file list is the set of engine files this repository's benchmark
# measurements depend on. It is deliberately explicit rather than a glob: a
# glob would silently stop covering a file that was renamed.
set -euo pipefail

HOST=${CANNS_BENCH_HOST:-222.29.101.57}
PORT=${CANNS_BENCH_PORT:-1693}
KEY=${CANNS_BENCH_KEY:-$HOME/.ssh/sichaohe_key}
# this script lives in `drivers/`, so the tree it syncs is its parent, and the
# repository root is two levels above that.
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
DST=${CANNS_BENCH_DIR:-/home/sichaohe/canns-lib-dev}

FILES="src/ripser/core/algorithm.rs src/ripser/core/assembly.rs \
src/ripser/core/reduction.rs src/ripser/matrix/dense.rs \
src/ripser/matrix/sparse.rs src/ripser/matrix/traits.rs \
Cargo.toml build.rs"

SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15 sichaohe@$HOST"

echo "syncing engine sources to $HOST:$PORT$DST"
for f in $FILES; do
    if [ ! -f "$REPO/$f" ]; then
        echo "  MISSING LOCALLY: $f" >&2
        exit 1
    fi
    local_size=$(stat -f%z "$REPO/$f" 2>/dev/null || stat -c%s "$REPO/$f")
    if [ "$local_size" -eq 0 ]; then
        echo "  REFUSING: $f is empty locally" >&2
        exit 1
    fi
    local_md5=$(md5 -q "$REPO/$f")
    $SSH "mkdir -p $(dirname "$DST/$f")"
    cat "$REPO/$f" | $SSH "cat > /tmp/_sync_one && mv /tmp/_sync_one $DST/$f"
    # The md5 and the size come back as two LINES, and `read` stops at the
    # first newline -- so the size would be silently empty. Join the lines
    # with a space first, then read them.
    remote_info=$($SSH "md5sum $DST/$f | cut -d' ' -f1; stat -c%s $DST/$f" | tr '\n' ' ')
    read -r remote_md5 remote_size <<<"$remote_info"
    if [ -z "$remote_size" ]; then
        echo "  REFUSING: $f -- the size of the copy on the host could not be" >&2
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
    printf "  ok  %-40s %8s bytes  %s\n" "$f" "$remote_size" "$remote_md5"
done
echo "all $(( $(echo $FILES | wc -w) )) engine files synced and verified"
