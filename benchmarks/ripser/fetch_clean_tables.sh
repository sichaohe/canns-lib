#!/bin/bash
# Fetch the clean tables from the benchmark host and check every one of
# them, with the quiet-window rule applied.
#
# THE GATE IS THE FOREIGN LOAD, NOT THE LOAD AVERAGE. The load average
# counts our own measurement -- the reduction runs on the host with one
# thread per core, and our own runs put the load average at 3.1 on this
# 16-core box -- so a threshold on it has to clear our own footprint, and
# then a neighbour's two-core job hides underneath it. Both directions
# were observed: a threshold of 1.5 discarded a batch whose busiest
# reading was 2.73, and a threshold of 3.0 would not have noticed a
# two-core neighbour at all. So every table carries, on every row, the
# number of cores used by processes that are NOT ours, and the checker
# requires it to be below half a core. Ours is excluded by construction,
# so the number needs no allowance for how busy the experiment is.
#
# The tables produced in a quiet window must PROVE it: every checker is run
# with --max-foreign-cpu 0.5, so a table with a row measured on a busy host
# is rejected rather than accepted with a footnote. A table that the host
# marked as contaminated is fetched too, and is expected to fail that
# check -- that is the test working.
set -uo pipefail

HOST=${CANNS_BENCH_HOST:-222.29.101.57}
PORT=${CANNS_BENCH_PORT:-1693}
KEY=${CANNS_BENCH_KEY:-$HOME/.ssh/sichaohe_key}
DELIV=${CANNS_DELIV:-/Volumes/home/projects/ph_work}
REPO=${CANNS_REPO:-/Volumes/home/projects/canns-lib/benchmarks/ripser}
SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15 sichaohe@$HOST"

# table : remote file : local name : checker : extra args
TABLES=(
  "scale_scan:/tmp/scale_clean.csv:scale_scan.csv:check_scale_scan.py:"
  "unified_ablation:/tmp/unified_ablation_clean.csv:unified_ablation.csv:check_unified_ablation.py:"
  "standard_bench:/tmp/standard_bench_5rep_clean.csv:standard_bench_5rep.csv:check_standard_bench.py:"
  "std_cpu:/tmp/standard_bench_ripser_cpu_clean.csv:standard_bench_ripser_cpu.csv:check_standard_bench.py:--min-reps 1"
  "contention:/tmp/contention_sweep_clean.csv:contention_sweep.csv:check_contention_sweep.py:"
)

echo "=== the host's own account of the run ==="
$SSH 'tail -14 /tmp/quiet_watch.log' 2>&1
echo

for t in "${TABLES[@]}"; do
    IFS=: read -r tag remote local checker extra <<<"$t"
    if ! $SSH "test -s $remote" 2>/dev/null; then
        echo "--- $tag: NOT PRODUCED ($remote)"
        continue
    fi
    scp -i "$KEY" -P "$PORT" "sichaohe@$HOST:$remote" "$DELIV/$local" >/dev/null 2>&1
    rows=$(($(wc -l < "$DELIV/$local") - 1))
    echo "--- $tag: $rows rows -> $DELIV/$local"
    python3 "$REPO/$checker" "$DELIV/$local" --max-foreign-cpu 0.5 $extra 2>&1 | tail -6
    echo
done

echo "=== the two independent copies of the scale scan ==="
# The same table measured twice on a quiet host: a neighbour's job would
# not make two independent measurements agree. This is the second line of
# evidence, and it is independent of the host's own accounting.
for f in /tmp/scale_batchA_1922.csv /tmp/scale_batchB_1849.csv; do
    $SSH "test -s $f" 2>/dev/null || continue
    scp -i "$KEY" -P "$PORT" "sichaohe@$HOST:$f" "$DELIV/$(basename $f)" >/dev/null 2>&1
    echo "--- fetched $(basename $f)"
done
if [ -s "$DELIV/scale_batchA_1922.csv" ] && [ -s "$DELIV/scale_clean.csv" ]; then
    python3 "$REPO/cross_batch.py" \
        "$DELIV/scale_batchA_1922.csv" "$DELIV/scale_clean.csv" \
        --tolerance 0.05 --out "$DELIV/cross_batch.csv" 2>&1 | tail -4
fi

echo "=== any batch the host marked as contaminated ==="
$SSH 'ls -1 /tmp/*.contaminated 2>/dev/null || echo "none"' 2>&1
