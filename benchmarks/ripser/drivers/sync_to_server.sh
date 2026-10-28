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
#
# A DIRECTORY is expanded here, file by file, and every file is still
# checked for size and md5 after the move. The point of the explicit list is
# that a file which did not land fails LOUDLY; a directory expansion keeps
# that property without a thirty-line list that has to be edited every time
# a checker is added, and a file that no longer exists is still noticed
# (the expansion would find nothing, and the count would not match).
#
# DOT FILES ARE SKIPPED. A macOS checkout carries `._name` AppleDouble files
# next to the real ones, and `find -name "*.py"` matches them: they are
# binary resource forks, they are not python, and one of them made the
# child-script check fail with a decoding error on a machine whose locale is
# not UTF-8.
FILES="harness/scale_scan.py harness/standard_bench.py harness/unified_ablation.py \
harness/t_arm_equivalence.py harness/foreign_cpu.py harness/o3_profile.py \
harness/o3_cpu_scaling.py harness/clean_h2h.py harness/h2h_coeff_ab.py \
harness/cofacet_early_exit_ab.py harness/barcode_equivalence.py \
harness/bench_core.py harness/corpus_arms.py harness/filtration_sweep.py \
harness/nsfix_ab.py \
harness/characterize_gridcells.py harness/validate_gridcells.py \
harness/bounds_gridcells_ab.py \
drivers/quiet_window_watch.sh drivers/corpus_quiet_run.sh \
drivers/build_abl_variants.sh drivers/build_measure_sites.sh \
drivers/measure_chain.sh \
drivers/fetch_clean_tables.sh drivers/sync_engine_to_server.sh \
tools/cross_batch.py tools/reduce_split.py tools/smoke_children.py"

# Directories to expand, one file per line, relative to the tree.
DIRS="harness tools patches archive"

SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15 sichaohe@$HOST"

echo "syncing to $HOST:$PORT$DST"
# Expand the directories into the file list, and drop the duplicates: the
# explicit list and the directories overlap on purpose, and a file must not
# be synced twice.
SYNC_FILES=""
for f in $FILES; do
    SYNC_FILES="$SYNC_FILES $f"
done
for d in $DIRS; do
    for f in $(cd "$SRC" && find "$d" -type f -name "*.py" ! -name "._*" | sort); do
        SYNC_FILES="$SYNC_FILES $f"
    done
done
SYNC_FILES=$(for f in $SYNC_FILES; do echo "$f"; done | sort -u)

for f in $SYNC_FILES; do
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
echo "all $(( $(echo $SYNC_FILES | wc -w) )) files synced and verified"

# THE FILES THAT NO LONGER EXIST HERE ARE DELETED THERE.
#
# The host's tree is not a git checkout, so a file that was deleted here
# stays there until it is removed explicitly -- and a stale copy is worse
# than a missing one: it is a program that still runs. The set is computed
# as "what is on the host and not here", never as a hard-coded list of names,
# so a file that is renamed or dropped is caught whichever way it happened.
#
# The path is quoted for the REMOTE shell, because it comes from the host's
# own directory listing.
q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# BOTH LISTS ARE SORTED IN THE C LOCALE. `comm` is a binary search over two
# sorted files, so the two files have to be sorted the SAME way: the laptop
# and the host sort in different locales by default, and with two different
# collations `comm` pairs up the first line and then stops -- so the
# cleanup silently removed one file out of forty-six.
#
# THE REMOTE LISTING DOES NOT SKIP THE DOT FILES. A macOS checkout carries
# `._name` AppleDouble files next to the real ones, and the sync copied them
# to the host as if they were python; they are binary resource forks, they
# are not python, and one of them made the child-script check fail with a
# decoding error on a machine whose locale is not UTF-8. They are not here,
# so the cleanup removes them.
(cd "$SRC" && find harness drivers tools patches archive -type f ! -name '._*' \
    | LC_ALL=C sort) > /tmp/_sync_local_files.txt
$SSH "cd $(q "$DST") && find . -type f ! -path '*/__pycache__/*' | sed 's|^\./||' | LC_ALL=C sort" \
    > /tmp/_sync_remote_files.txt
removed=0
while IFS= read -r rel; do
    [ -z "$rel" ] && continue
    echo "  removing $rel (it no longer exists here)"
    # STDIN IS CLOSED FOR THE SSH. `ssh` forwards its standard input to the
    # remote host, and this loop is reading the list of files to remove from
    # its own standard input -- so the first ssh ate the rest of the list and
    # the cleanup removed exactly one file, every time.
    $SSH "rm -f -- $(q "$DST/$rel")" < /dev/null
    removed=$((removed + 1))
done < <(LC_ALL=C comm -13 /tmp/_sync_local_files.txt /tmp/_sync_remote_files.txt)

# THE BYTECODE CACHES GO TOO. They are regenerated on the next run and they
# are not tracked anywhere, so a stale one is only ever a trap.
$SSH "cd $(q "$DST") && find . -type d -name __pycache__ -prune -exec rm -rf {} +" < /dev/null
if [ "$removed" -eq 0 ]; then
    echo "  nothing to remove: the host's tree and this one agree"
else
    echo "  removed $removed file(s) from the host"
fi
