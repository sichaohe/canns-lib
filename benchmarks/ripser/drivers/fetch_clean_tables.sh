#!/bin/bash
# Fetch the tables from the benchmark host into the results project, and say
# for each one whether it can be shown to have been measured on a quiet host.
#
# THE GATE IS THE FOREIGN LOAD, NOT THE LOAD AVERAGE. The load average counts
# our own measurement -- the reduction runs on the host with one thread per
# core, and our own runs put the load average at 3.1 on this 16-core box --
# so a threshold on it has to clear our own footprint, and then a neighbour's
# two-core job hides underneath it. Both directions were observed: a threshold
# of 1.5 discarded a batch whose busiest reading was 2.73, and a threshold of
# 3.0 would not have noticed a two-core neighbour at all. So every table
# carries, on every row, the number of cores used by processes that are NOT
# ours, and the gate is a QUARTER of a core. Ours is excluded by construction,
# so the number needs no allowance for how busy the experiment is.
#
# The gate itself lives in `harness/bench_core.py`, so this script and the
# drivers that produce the tables ask the same question the same way.
#
# A table the host marked as contaminated is fetched too, and is expected to
# fail the gate -- that is the gate working.
set -uo pipefail

HOST=${CANNS_BENCH_HOST:-222.29.101.57}
PORT=${CANNS_BENCH_PORT:-1693}
KEY=${CANNS_BENCH_KEY:-$HOME/.ssh/sichaohe_key}
DELIV=${CANNS_DELIV:-/Volumes/home/projects/ph_work}
REPO=${CANNS_REPO:-/Volumes/home/projects/canns-lib/benchmarks/ripser}
MODULE=$REPO/harness
SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15 sichaohe@$HOST"

# table : remote file : local name
TABLES=(
  "scale_scan:/tmp/scale_clean.csv:scale_scan.csv"
  "unified_ablation:/tmp/unified_ablation_clean.csv:unified_ablation.csv"
  "standard_bench:/tmp/standard_bench_5rep_clean.csv:standard_bench_5rep.csv"
  "std_cpu:/tmp/standard_bench_ripser_cpu_clean.csv:standard_bench_ripser_cpu.csv"
  "clean_h2h:/tmp/clean_h2h_clean.csv:clean_h2h.csv"
  "filtration_sweep:/tmp/filtration_sweep_clean.csv:filtration_sweep.csv"
  "o3_profile:/tmp/o3_profile_clean.csv:o3_profile.csv"
  "o3_cpu_scaling:/tmp/o3_cpu_scaling_clean.csv:o3_cpu_scaling.csv"
  "early_exit_ab:/tmp/cofacet_early_exit_ab_clean.csv:cofacet_early_exit_ab.csv"
  "characterize_gridcells:/tmp/characterize_gridcells_clean.csv:characterize_gridcells.csv"
  "gridcells_validation:/tmp/gridcells_validation_clean.csv:gridcells_validation.csv"
)

echo "=== the host's own account of the run ==="
$SSH 'tail -14 /tmp/quiet_watch.log' 2>&1
echo

for t in "${TABLES[@]}"; do
    IFS=: read -r tag remote local <<<"$t"
    if ! $SSH "test -s $remote" 2>/dev/null; then
        echo "--- $tag: NOT PRODUCED ($remote)"
        continue
    fi
    scp -i "$KEY" -P "$PORT" "sichaohe@$HOST:$remote" "$DELIV/$local" >/dev/null 2>&1
    rows=$(($(wc -l < "$DELIV/$local") - 1))
    printf -- "--- %-22s %4s rows -> %s\n" "$tag" "$rows" "$DELIV/$local"
    python3 - "$DELIV/$local" "$MODULE" <<'PY'
import sys
path, module = sys.argv[1], sys.argv[2]
sys.path.insert(0, module)
import bench_core
worst, missing, broken, timed_out, rows = bench_core.foreign_summary(path)
print("      worst foreign load: mean %.3f cores, %d row(s) without a reading, "
      "%d broken, %d timed out, %d rows"
      % (worst, missing, broken, timed_out, rows))
if missing or broken or worst > bench_core.FOREIGN_GATE:
    print("      NOT A MEASUREMENT ON A QUIET HOST -- do not quote this table")
else:
    print("      measured on a quiet host")
PY
done

echo
echo "=== the independent copies of the scale scan ==="
# The same table measured more than once on a quiet host: a neighbour's job
# would not make independent measurements agree. This is the second line of
# evidence, and it is independent of the host's own accounting.
#
# NOTE THE FILE NAMES. The current batch is fetched by the loop above and
# lands under the local name `scale_scan.csv`; the two earlier copies keep the
# names they were produced under.
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
    # ON THE PROCESS CLOCK. The library clock is missing from one of the
    # earlier copies -- that copy predates the column -- and a schema
    # difference is not a disagreement about the numbers, so comparing it
    # would report a failure that is not one. The process clock is in every
    # copy.
    python3 "$REPO/tools/cross_batch.py" "${COPIES[@]}" \
        --tolerance 0.05 --columns wall_s --out "$DELIV/cross_batch.csv" 2>&1 | tail -4
else
    echo "--- only ${#COPIES[@]} copy/copies of the scale scan: nothing to compare"
fi

echo
echo "=== the corpus table ==="
# The 97 matrices are not redistributable yet (the licence question is
# open), so this table is NOT published -- it is fetched here for the
# experiment record, and it stays in the results project.
if $SSH "test -s /tmp/gca_results_clean.csv" 2>/dev/null; then
    scp -i "$KEY" -P "$PORT" "sichaohe@$HOST:/tmp/gca_results_clean.csv" \
        "$DELIV/gca_results_clean.csv" >/dev/null 2>&1
    echo "--- fetched gca_results_clean.csv ($(( $(wc -l < "$DELIV/gca_results_clean.csv") - 1 )) rows)"
else
    echo "--- corpus table NOT PRODUCED"
fi

echo
echo "=== any batch the host marked as contaminated ==="
$SSH 'ls -1 /tmp/*.contaminated 2>/dev/null || echo "none"' 2>&1
