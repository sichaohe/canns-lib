#!/bin/bash
# Fetch the clean tables from the benchmark host and check every one of
# them, with the quiet-window rule applied.
#
# The threshold is 3.0, not 1.5: the load average includes our own
# measurement, whose footprint on this 16-core host is up to 2.7 cores,
# while a neighbour's job was 7-9 cores and put the load at 9-19.
#
# The tables produced in a quiet window must PROVE it: every checker is run
# with --max-loadavg 3.0, so a table with a row measured on a busy host is
# rejected rather than accepted with a footnote. A table that the host
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
    python3 "$REPO/$checker" "$DELIV/$local" --max-loadavg 3.0 $extra 2>&1 | tail -6
    echo
done

echo "=== any batch the host marked as contaminated ==="
$SSH 'ls -1 /tmp/*.contaminated 2>/dev/null || echo "none"' 2>&1
