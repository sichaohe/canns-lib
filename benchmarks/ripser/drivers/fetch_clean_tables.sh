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
# with --max-foreign-cpu 0.25, so a table with a row measured on a busy host
# is rejected rather than accepted with a footnote. A table that the host
# marked as contaminated is fetched too, and is expected to fail that
# check -- that is the test working.
#
# The gate is a QUARTER of a core, not half of one. Half a core was the
# value this script used before the gate was pinned down; it is twice as
# loose as the threshold every checker defaults to, so a table that the
# checkers reject on the server would have been accepted here. The number
# in the flag and the number in the checker must be the same number.
set -uo pipefail

HOST=${CANNS_BENCH_HOST:-222.29.101.57}
PORT=${CANNS_BENCH_PORT:-1693}
KEY=${CANNS_BENCH_KEY:-$HOME/.ssh/sichaohe_key}
DELIV=${CANNS_DELIV:-/Volumes/home/projects/ph_work}
REPO=${CANNS_REPO:-/Volumes/home/projects/canns-lib/benchmarks/ripser}
SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15 sichaohe@$HOST"

# table : remote file : local name : checker : extra args
TABLES=(
  "scale_scan:/tmp/scale_clean.csv:scale_scan.csv:checkers/check_scale_scan.py:"
  "unified_ablation:/tmp/unified_ablation_clean.csv:unified_ablation.csv:checkers/check_unified_ablation.py:"
  "standard_bench:/tmp/standard_bench_5rep_clean.csv:standard_bench_5rep.csv:checkers/check_standard_bench.py:"
  "std_cpu:/tmp/standard_bench_ripser_cpu_clean.csv:standard_bench_ripser_cpu.csv:checkers/check_standard_bench.py:--min-reps 1 --sides ripser_cpu --reference-side ripser_cpu"
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
    python3 "$REPO/$checker" "$DELIV/$local" --max-foreign-cpu 0.25 $extra 2>&1 | tail -6
    echo
done

echo "=== the independent copies of the scale scan ==="
# The same table measured more than once on a quiet host: a neighbour's job
# would not make independent measurements agree. This is the second line of
# evidence, and it is independent of the host's own accounting.
#
# NOTE THE FILE NAMES. Batch C is fetched by the table loop above and lands
# under the local name `scale_scan.csv`; the two earlier copies keep the
# names they were produced under. The comparison below used to look for
# `scale_clean.csv`, a name nothing ever writes, so the whole step was
# skipped in silence and the cross-batch table was never regenerated.
COPIES=()
for f in /tmp/scale_batchA_1922.csv /tmp/scale_batchB_1849.csv; do
    $SSH "test -s $f" 2>/dev/null || continue
    scp -i "$KEY" -P "$PORT" "sichaohe@$HOST:$f" "$DELIV/$(basename $f)" >/dev/null 2>&1
    echo "--- fetched $(basename $f)"
done
for local in scale_batchA_1922.csv scale_batchB_1849.csv scale_scan.csv; do
    [ -s "$DELIV/$local" ] && COPIES+=("$DELIV/$local")
done
if [ "${#COPIES[@]}" -ge 2 ]; then
    echo "--- comparing ${#COPIES[@]} independent copies of the scale scan"
    python3 "$REPO/tools/cross_batch.py" "${COPIES[@]}" \
        --tolerance 0.05 --out "$DELIV/cross_batch.csv" 2>&1 | tail -4
else
    echo "--- only ${#COPIES[@]} copy/copies of the scale scan: nothing to compare"
fi

echo "=== the corpus table (its checker needs the corpus, so it runs there) ==="
# The 97 matrices are not redistributable yet (the licence question is
# open), so this table is NOT published -- it is fetched here for the
# paper's numbers, and the checker is run on the host where the corpus is.
if $SSH "test -s /tmp/gca_results_clean.csv" 2>/dev/null; then
    scp -i "$KEY" -P "$PORT" "sichaohe@$HOST:/tmp/gca_results_clean.csv" \
        "$DELIV/gca_results_clean.csv" >/dev/null 2>&1
    echo "--- fetched gca_results_clean.csv ($(( $(wc -l < "$DELIV/gca_results_clean.csv") - 1 )) rows)"
    $SSH 'cd ~/canns-lib-dev && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl && python benchmarks/ripser/checkers/check_gridcells_bench.py /tmp/gca_results_clean.csv --max-foreign-cpu 0.25 --self-test' 2>&1 | tail -14
else
    echo "--- corpus table NOT PRODUCED"
fi

echo "=== any batch the host marked as contaminated ==="
$SSH 'ls -1 /tmp/*.contaminated 2>/dev/null || echo "none"' 2>&1
