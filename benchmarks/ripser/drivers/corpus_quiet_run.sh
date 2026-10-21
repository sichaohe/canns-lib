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
MODULE=$REPO/benchmarks/ripser

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
    python3 - "$1" "$MAX_FOREIGN" <<'PY'
import csv, sys
path, limit = sys.argv[1], float(sys.argv[2])
worst, missing = 0.0, 0
try:
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            raw = (r.get("foreign_cpu") or "").strip()
            if not raw:
                missing += 1
                continue
            try:
                worst = max(worst, float(raw))
            except ValueError:
                missing += 1
except OSError:
    print("no table to check")
    sys.exit(2)
print("worst foreign load in the batch: mean %.3f cores (%d row(s) without "
      "a reading)" % (worst, missing))
if missing:
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
            # the checker needs the corpus, so it runs where the corpus is
            python3 benchmarks/ripser/checkers/check_gridcells_bench.py "$OUT" \
                --max-foreign-cpu "$MAX_FOREIGN" --self-test \
                > /tmp/gca_clean_check.log 2>&1
            echo "$(date +%Y-%m-%dT%H:%M:%S) checked: $OUT" >> "$LOG"
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
        for spec in \
            "characterize_gridcells:/tmp/characterize_gridcells_clean.csv:benchmarks/ripser/harness/characterize_gridcells.py --site /home/sichaohe/canns-lib-dev/site_ship6 --out /tmp/characterize_gridcells_clean.csv:benchmarks/ripser/checkers/check_characterize_gridcells.py" \
            "gridcells_validation:/tmp/gridcells_validation_clean.csv:benchmarks/ripser/harness/validate_gridcells.py --site /home/sichaohe/canns-lib-dev/site_cpu_final --reps 2 --out /tmp/gridcells_validation_clean.csv:benchmarks/ripser/checkers/check_validate_gridcells.py"
        do
            IFS=: read -r tag out cmd checker <<<"$spec"
            rm -f "$out"
            python3 $cmd > "/tmp/${tag}_clean.log" 2>&1
            if batch_is_clean "$out"; then
                echo "$(date +%Y-%m-%dT%H:%M:%S) clean: $out" >> "$LOG"
                python3 "$checker" "$out" --max-foreign-cpu "$MAX_FOREIGN" \
                    > "/tmp/${tag}_clean_check.log" 2>&1
                echo "$(date +%Y-%m-%dT%H:%M:%S) checked: $out" >> "$LOG"
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
