#!/bin/bash
# Re-measure the grid-cells corpus benchmark on a quiet host, with the
# foreign-load reading recorded on every cell.
#
# WHY IT IS RE-MEASURED. The corpus table (582 cells) is the source of the
# paper's "0.620x against the fair ripser++ arm" number, and it was
# measured before the tables carried a foreign-load reading. What it did
# carry was the load average -- and the ratio this table reports MOVES with
# it: measured on the old table, the median ratio is 0.617 for the cells
# whose load average was below 1 and 0.674 for the cells whose load average
# was 3-4. The load average cannot say whether that load was ours (the
# corpus table includes a CPU arm that uses every core) or a neighbour's,
# so the old table cannot show it was measured on a quiet host.
#
# The driver is the corpus' own (gca_bench.py), patched to record the
# foreign load per cell. It is not in this repository: it belongs to the
# corpus pipeline, whose licence is still open (see PAPER-SECTION-ARTIFACT
# 8.3). The tables it produces are therefore NOT published either.
#
# A BATCH IS ALL OR NOTHING: if any cell was measured while somebody else
# was using the host, the whole table is moved aside and the window is
# waited for again.
# NOT `set -u`: conda's activate script reads ADDR2LINE, which is not
# set in a non-interactive shell, and `set -u` turns that into a
# fatal error that kills this script before it measures anything.
set -eo pipefail

LOG=/tmp/corpus_watch.log
REPO=${CANNS_BENCH_REPO:-/home/sichaohe/canns-lib-dev}
DRIVER=${GCA_DRIVER:-/home/sichaohe/ph_work/gcbench/gca_bench.py}
OUT=/tmp/gca_results_clean.csv
# The probe lives in `harness/`, NEXT TO the runners that import it. This
# path was the tree root, which was correct until the harness became a
# tree; after the move it was wrong, and the probe failed with an import
# error that `2>/dev/null` swallowed -- so the watcher read an EMPTY
# foreign load every minute and, because an empty reading is not a
# quiet one, it never started. The symptom was a watcher that logged
# `foreign_cores=` forever.
MODULE=$REPO/benchmarks/ripser/harness

MAX_FOREIGN=0.25
QUIET_FOREIGN=0.25

cd "$REPO" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl

foreign_now() {
    python3 -c "
import sys
sys.path.insert(0, '$MODULE')
import foreign_cpu
print('%.3f' % foreign_cpu.foreign_cores(1.0))
" 2>/dev/null || echo ""
}

batch_is_clean() {
    # The gate lives in bench_core: one implementation, used by every driver.
    python3 - "$1" "$MAX_FOREIGN" "$MODULE" <<'PY'
import sys
path, limit, module = sys.argv[1], float(sys.argv[2]), sys.argv[3]
sys.path.insert(0, module)
import bench_core
try:
    worst, missing, broken, timed_out, rows = bench_core.foreign_summary(path)
except OSError:
    print("no table to check")
    sys.exit(2)
print("worst foreign load in the batch: mean %.3f cores (%d row(s) without a "
      "reading, %d broken row(s), %d timed out, %d rows)"
      % (worst, missing, broken, timed_out, rows))
if broken:
    print("rows whose run did not finish: %d -- this is a BUG, not a busy host"
          % broken)
if missing:
    print("rows without a foreign-load reading: %d" % missing)
if timed_out:
    print("rows that hit the time limit: %d -- a legal result, recorded as such"
          % timed_out)
if broken or missing:
    sys.exit(1)
sys.exit(0 if worst <= limit else 1)
PY
}

quiet=0
for i in $(seq 1 1440); do
    fc=$(foreign_now)
    if [ -n "$fc" ] && awk -v f="$fc" -v m="$QUIET_FOREIGN" 'BEGIN{exit !(f<m)}'; then
        quiet=$((quiet + 1))
    else
        quiet=0
    fi
    echo "$(date +%Y-%m-%dT%H:%M:%S) foreign_cores=$fc quiet_streak=$quiet" >> "$LOG"
    if [ "$quiet" -ge 3 ]; then
        echo "$(date +%Y-%m-%dT%H:%M:%S) window is quiet -- running the corpus benchmark" >> "$LOG"
        rm -f "$OUT"
        GCA_OUT="$OUT" python3 "$DRIVER" > /tmp/gca_clean.log 2>&1
        if batch_is_clean "$OUT"; then
            echo "$(date +%Y-%m-%dT%H:%M:%S) clean: $OUT" >> "$LOG"
        else
            mv "$OUT" "$OUT.contaminated"
            echo "$(date +%Y-%m-%dT%H:%M:%S) CONTAMINATED: $OUT -> $OUT.contaminated" >> "$LOG"
        fi

        # THE OTHER TWO CORPUS TABLES. The characterisation and the
        # per-cell barcode cross-check run over the same 97 matrices, they
        # both quote timings, and neither carried a foreign-load reading
        # before the gate existed -- so neither could be shown to have been
        # measured on a quiet host. They are measured in the SAME window as
        # the corpus benchmark, which is the point: the window is the scarce
        # resource, and three tables that disagree about how busy the host
        # was are worse than one that agrees.
        #
        # ONE FAILURE DOES NOT KILL THE REST. A step that fails is recorded
        # in the log and the chain goes on to the next one: a checker that
        # returned non-zero used to take the whole chain down with it under
        # `set -e`, so the two tables after it were never produced.
        for spec in \
            "characterize_gridcells:/tmp/characterize_gridcells_clean.csv:benchmarks/ripser/harness/characterize_gridcells.py --site /home/sichaohe/canns-lib-dev/site_ship6 --out /tmp/characterize_gridcells_clean.csv" \
            "gridcells_validation:/tmp/gridcells_validation_clean.csv:benchmarks/ripser/harness/validate_gridcells.py --site /home/sichaohe/canns-lib-dev/site_cpu_final --reps 2 --out /tmp/gridcells_validation_clean.csv"
        do
            IFS=: read -r tag out cmd <<<"$spec"
            rm -f "$out"
            python3 $cmd > "/tmp/${tag}_clean.log" 2>&1
            if batch_is_clean "$out"; then
                echo "$(date +%Y-%m-%dT%H:%M:%S) clean: $out" >> "$LOG"
            else
                mv "$out" "$out.contaminated"
                echo "$(date +%Y-%m-%dT%H:%M:%S) CONTAMINATED: $out -> $out.contaminated" >> "$LOG"
            fi
        done
        exit 0
    fi
    sleep 60
done
echo "$(date +%Y-%m-%dT%H:%M:%S) gave up waiting for a quiet window" >> "$LOG"
exit 1
