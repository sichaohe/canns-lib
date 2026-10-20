#!/bin/bash
# Wait for a quiet window on this shared box, then measure.
#
# The box has another user on it, and a concurrent job changes the numbers
# by up to 40% -- and not always in the same direction (the ratio of our
# time to the reference's flipped sign under load). So a measurement taken
# now would not be a measurement of the code.
#
# The window condition: the load average below 1.5 for three consecutive
# checks, a minute apart.
#
# A BATCH IS ALL OR NOTHING. Checking the load once a minute is not
# enough: a neighbour's job can start in the middle of a measurement, and
# half a table measured under load is worse than no table, because it
# looks like a table. So after every measurement the load average on
# EVERY row of the table it produced is checked, and a batch with a
# contaminated row is moved aside (not deleted) and the window is waited
# for again.
#
# The measurements, in order: the scale scan; the ablation variants and
# the unified ablation table; the standard suite with five repetitions;
# the reference CPU baseline with a 900 s limit.
#
# Everything is written to /tmp; nothing is deleted.

LOG=/tmp/quiet_watch.log
MAX_LOAD=1.5

cd ~/canns-lib-dev || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl

batch_is_clean() {
    python3 - "$1" "$MAX_LOAD" <<'PY'
import csv, sys
path, limit = sys.argv[1], float(sys.argv[2])
worst = 0.0
try:
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                worst = max(worst, float(r.get("loadavg") or 0.0))
            except ValueError:
                pass
except OSError:
    sys.exit(1)
print("worst loadavg in the batch: %.2f" % worst)
sys.exit(0 if worst < limit else 1)
PY
}

measure() {
    out=$1; log=$2; shift 2
    rm -f "$out"
    "$@" > "$log" 2>&1
    if batch_is_clean "$out"; then
        echo "$(date +%Y-%m-%dT%H:%M:%S) clean: $out" >> "$LOG"
        return 0
    fi
    mv "$out" "$out.contaminated"
    echo "$(date +%Y-%m-%dT%H:%M:%S) CONTAMINATED: $out -> $out.contaminated" >> "$LOG"
    return 1
}

quiet=0
for i in $(seq 1 1440); do
    l=$(cut -d" " -f1 /proc/loadavg)
    if awk -v l="$l" -v m="$MAX_LOAD" 'BEGIN{exit !(l<m)}'; then
        quiet=$((quiet + 1))
    else
        quiet=0
    fi
    echo "$(date +%Y-%m-%dT%H:%M:%S) loadavg=$l quiet_streak=$quiet" >> "$LOG"
    if [ "$quiet" -ge 3 ]; then
        echo "$(date +%Y-%m-%dT%H:%M:%S) window is quiet -- starting" >> "$LOG"
        if measure /tmp/scale_clean.csv /tmp/scale_clean.log \
             python benchmarks/ripser/scale_scan.py --sizes 1200,1600,2000 \
             --timeout 600 --out /tmp/scale_clean.csv; then
            bash benchmarks/ripser/build_abl_variants.sh > /tmp/abl_variants_build.log 2>&1
            echo "$(date +%Y-%m-%dT%H:%M:%S) ablation variants built" >> "$LOG"
            measure /tmp/unified_ablation_clean.csv /tmp/unified_ablation_clean.log \
                python benchmarks/ripser/unified_ablation.py --reps 3 \
                --out /tmp/unified_ablation_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) unified ablation done" >> "$LOG"
            measure /tmp/standard_bench_5rep_clean.csv /tmp/standard_bench_5rep_clean.log \
                python benchmarks/ripser/standard_bench.py --sides pp_gpu,gpu,cpu \
                --reps 5 --timeout 600 --out /tmp/standard_bench_5rep_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) standard suite (5 reps) done" >> "$LOG"
            measure /tmp/standard_bench_ripser_cpu_clean.csv /tmp/standard_bench_ripser_cpu_clean.log \
                python benchmarks/ripser/standard_bench.py --sides ripser_cpu \
                --reps 2 --timeout 900 --out /tmp/standard_bench_ripser_cpu_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) reference CPU baseline done" >> "$LOG"
        else
            echo "$(date +%Y-%m-%dT%H:%M:%S) the scale scan was contaminated -- waiting for another window" >> "$LOG"
        fi
        exit 0
    fi
    sleep 60
done
echo "$(date +%Y-%m-%dT%H:%M:%S) gave up waiting for a quiet window" >> "$LOG"
exit 1
