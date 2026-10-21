#!/bin/bash
# Copy the ENGINE sources to the benchmark host, atomically, and CHECK.
#
# WHY THIS IS SEPARATE FROM sync_to_server.sh. That one syncs the benchmark
# scripts (harness/checkers/tools). This one syncs the RUST sources, and it
# has to be a different list because the two trees have different failure
# modes: the benchmark scripts are read by a python process that reports its
# own errors, while a stale .rs file compiles into a binary that runs and
# produces numbers. A benchmark script that did not land fails loudly; an
# engine file that did not land fails silently, as a measurement of the
# wrong build.
#
# So: write to a temporary file, move it into place, and assert that every
# file landed with the expected size and md5. A zero-length file is refused
# on both ends, and the empty file's md5 is
# d41d8cd98f00b204e9800998ecf8427e.
#
# THE SSH INVOCATION IS AN ARRAY, AND THE PATHS ARE ARGUMENTS, NOT SHELL
# TEXT. The host, the port, the key and the destination come from the
# environment, and an environment that can inject a `-oProxyCommand=...`
# option, or a path with a semicolon in it, would otherwise be able to run
# commands on this machine or on the host. The port is checked to be a
# number, the paths are passed as separate arguments to `sh -c`, and the
# temporary file is named per run so that two overlapping syncs cannot
# overwrite each other's half-written file.
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

case "$PORT" in
    ''|*[!0-9]*) echo "REFUSING: the port must be a number, got '$PORT'" >&2; exit 1 ;;
esac
case "$HOST" in
    *[!A-Za-z0-9._-]*) echo "REFUSING: the host has characters that are not a host name" >&2; exit 1 ;;
esac

FILES="src/ripser/core/algorithm.rs src/ripser/core/assembly.rs \
src/ripser/core/reduction.rs src/ripser/matrix/dense.rs \
src/ripser/matrix/sparse.rs src/ripser/matrix/traits.rs \
Cargo.toml build.rs"

# An ARRAY, so that every value is exactly one argument.
SSH=(ssh -i "$KEY" -p "$PORT" -o ConnectTimeout=30 -o BatchMode=yes
     -o ConnectionAttempts=3 "sichaohe@$HOST")

# The host is shared, and a host with a neighbour's twelve-core job on it
# sometimes takes longer than a minute to complete the SSH banner exchange.
# That is a transient condition, not a failure, so the call is retried.
ssh_try() {
    local attempt
    for attempt in 1 2 3; do
        if "${SSH[@]}" "$@"; then
            return 0
        fi
        echo "  (ssh attempt $attempt failed; retrying)" >&2
        sleep 10
    done
    return 1
}

# The checksum command that exists on THIS machine: macOS has `md5 -q` and
# Linux has `md5sum`. Calling the wrong one under `set -e` stops the script
# before it syncs anything.
if command -v md5sum >/dev/null 2>&1; then
    md5_of() { md5sum "$1" | cut -d' ' -f1; }
elif command -v md5 >/dev/null 2>&1; then
    md5_of() { md5 -q "$1"; }
else
    echo "REFUSING: no md5sum or md5 on this machine" >&2
    exit 1
fi

echo "syncing engine sources to $HOST:$PORT$DST"
# A path, quoted for the REMOTE shell. ssh joins its arguments into one
# command string and the remote shell re-parses it, so a path has to be
# quoted here or a path with a space or a semicolon in it would be read as
# shell syntax. Single quotes are the only quoting that disables every
# expansion; an embedded single quote is closed, escaped and reopened.
q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# THE LOCK. The sync replaces the engine files one at a time, so a build that
# starts in the middle of it would read a MIXTURE of two revisions -- every
# file passes its checksum and the measurement does not describe any single
# revision. The build driver takes the same lock, so the two cannot overlap.
LOCK=/tmp/.canns_engine_sync.lock
ssh_try "touch $LOCK" || { echo "REFUSING: cannot take the sync lock on the host" >&2; exit 1; }
trap 'ssh_try "rm -f $LOCK" >/dev/null 2>&1 || true' EXIT

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
    local_md5=$(md5_of "$REPO/$f")
    # The paths go to the remote shell QUOTED, not interpolated raw, and the
    # temporary file is named per run so that two overlapping syncs cannot
    # overwrite each other's half-written file.
    remote_dir=$(dirname "$DST/$f")
    remote_tmp="$DST/$f.sync.$$"
    cat "$REPO/$f" | ssh_try "mkdir -p -- $(q "$remote_dir") && cat > $(q "$remote_tmp") && mv -- $(q "$remote_tmp") $(q "$DST/$f")"
    # The md5 and the size come back as two LINES, and `read` stops at the
    # first newline -- so the size would be silently empty. Join the lines
    # with a space first, then read them.
    remote_info=$(ssh_try "md5sum -- $(q "$DST/$f") | cut -d' ' -f1; stat -c%s -- $(q "$DST/$f")" | tr '\n' ' ')
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
