#!/bin/bash
# Wait for a quiet window on this shared box, then measure.
#
# The box has another user on it, and a concurrent job changes the numbers
# by up to 40% -- and not always in the same direction (the ratio of our
# time to the reference's flipped sign under load). So a measurement taken
# now would not be a measurement of the code.
#
# HOW THE BOX IS ASKED WHETHER IT IS QUIET. The load average cannot answer
# that. It counts our own measurement -- the reduction runs on the host with
# one thread per core, and our own runs put the load average at 3.1 on this
# 16-core host -- so a threshold on it has to clear our own footprint, and
# then a neighbour's two-core job hides underneath it. That is not
# hypothetical: a threshold of 3.0 discarded a batch whose worst reading was
# 2.73, and a threshold of 1.5 (which the running copy of this script still
# had in memory) discarded it again, for the same reason.
#
# So the box is asked directly. Every process's CPU time is read from /proc,
# the ones owned by another user are summed, and the answer is a number of
# CORES used by somebody else. Ours is excluded by construction, so the
# number needs no fudge factor for how busy we are. See foreign_cpu.py.
#
# A BATCH IS ALL OR NOTHING. Checking the box once a minute is not enough:
# a neighbour's job can start in the middle of a measurement, and half a
# table measured under load is worse than no table, because it looks like a
# table. So after every measurement the foreign load on EVERY row of the
# table it produced is checked, and a batch with a contaminated row is moved
# aside (not deleted) and the window is waited for again.
#
# The measurements, in order: the scale scan; the ablation variants and the
# unified ablation table; the standard suite with five repetitions; the
# reference CPU baseline.
#
# Everything is written to /tmp; nothing is deleted.

LOG=/tmp/quiet_watch.log
REPO=${CANNS_BENCH_REPO:-/home/sichaohe/canns-lib-dev}
MODULE=$REPO/benchmarks/ripser

# The gate. A row measured while somebody else was using more than this
# many cores ON AVERAGE is not a measurement of our code. The number is
# the MEAN foreign load over the row (the total CPU time other people
# used, divided by the row's wall time), not the largest burst: a
# neighbour's job of seven to nine cores shows up here as seven or nine,
# and a desktop compositor that used one core for half a second out of a
# two-minute measurement shows up as 0.004, which is what it is worth.
MAX_FOREIGN=0.25
# The same question, asked before the batch starts, to decide whether the
# box is quiet enough to start at all.
QUIET_FOREIGN=0.25

cd "$REPO" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl

foreign_now() {
    # The number of cores used by processes that are not ours, right now.
    python3 -c "
import sys
sys.path.insert(0, '$MODULE')
import foreign_cpu
print('%.3f' % foreign_cpu.foreign_cores(1.0))
" 2>/dev/null || echo ""
}

load_now() {
    cut -d" " -f1 /proc/loadavg
}

batch_is_clean() {
    # The batch's own evidence: the foreign load recorded on every row
    # while that row was being measured.
    python3 - "$1" "$MAX_FOREIGN" <<'PY'
import csv, sys
path, limit = sys.argv[1], float(sys.argv[2])
worst, worst_la, missing = 0.0, 0.0, 0
try:
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            raw = (r.get("foreign_cpu") or "").strip()
            if not raw:
                missing += 1
                continue
            try:
                worst = max(worst, float(raw))
                worst_la = max(worst_la, float(r.get("loadavg") or 0.0))
            except ValueError:
                missing += 1
except OSError:
    print("no table to check")
    sys.exit(2)
print("worst foreign load in the batch: mean %.3f cores (worst load average "
      "%.2f, %d row(s) without a reading)" % (worst, worst_la, missing))
if missing:
    print("rows without a foreign-load reading: %d" % missing)
    sys.exit(1)
sys.exit(0 if worst <= limit else 1)
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
    la=$(load_now)
    fc=$(foreign_now)
    if [ -n "$fc" ] && awk -v f="$fc" -v m="$QUIET_FOREIGN" 'BEGIN{exit !(f<m)}'; then
        quiet=$((quiet + 1))
    else
        quiet=0
    fi
    echo "$(date +%Y-%m-%dT%H:%M:%S) loadavg=$la foreign_cores=$fc quiet_streak=$quiet" >> "$LOG"
    if [ "$quiet" -ge 3 ]; then
        echo "$(date +%Y-%m-%dT%H:%M:%S) window is quiet -- starting" >> "$LOG"
        if measure /tmp/scale_clean.csv /tmp/scale_clean.log \
             python benchmarks/ripser/harness/scale_scan.py --sizes 1200,1600,2000 \
             --timeout 600 --out /tmp/scale_clean.csv; then
            bash benchmarks/ripser/drivers/build_abl_variants.sh > /tmp/abl_variants_build.log 2>&1
            echo "$(date +%Y-%m-%dT%H:%M:%S) ablation variants built" >> "$LOG"
            measure /tmp/unified_ablation_clean.csv /tmp/unified_ablation_clean.log \
                python benchmarks/ripser/harness/unified_ablation.py --reps 3 \
                --out /tmp/unified_ablation_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) unified ablation done" >> "$LOG"
            measure /tmp/standard_bench_5rep_clean.csv /tmp/standard_bench_5rep_clean.log \
                python benchmarks/ripser/harness/standard_bench.py --sides pp_gpu,gpu,cpu \
                --reps 5 --timeout 600 --out /tmp/standard_bench_5rep_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) standard suite (5 reps) done" >> "$LOG"
            measure /tmp/standard_bench_ripser_cpu_clean.csv /tmp/standard_bench_ripser_cpu_clean.log \
                python benchmarks/ripser/harness/standard_bench.py --sides ripser_cpu \
                --reps 2 --timeout 900 --out /tmp/standard_bench_ripser_cpu_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) reference CPU baseline done" >> "$LOG"

            # THE HEAD-TO-HEAD. This is the paper's central comparison, and it
            # was the table most exposed to the host's load: a neighbour's
            # job moved the ratio by 15% between two windows, which is why
            # the ratio is host-dependent and cannot be quoted from a
            # measurement that does not say how busy the host was.
            #
            # `VAR=x func` is NOT used here: in bash a variable assignment
            # before a FUNCTION call survives the call, so the next
            # measurement would silently inherit the previous one's site.
            export CANNS_RIPSER_SITE=$REPO/site_ship6
            measure /tmp/clean_h2h_clean.csv /tmp/clean_h2h_clean.log \
                python benchmarks/ripser/harness/clean_h2h.py --reps 9 \
                --timeout 900 --out /tmp/clean_h2h_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) head-to-head done" >> "$LOG"

            measure /tmp/filtration_sweep_clean.csv /tmp/filtration_sweep_clean.log \
                python benchmarks/ripser/harness/filtration_sweep.py --reps 2 \
                --out /tmp/filtration_sweep_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) filtration sweep done" >> "$LOG"
            unset CANNS_RIPSER_SITE

            # The early-exit A/B, on the CPU build -- the one the classical
            # suite's "our build against the single-threaded classic Ripser"
            # row is about.
            measure /tmp/cofacet_early_exit_ab_clean.csv /tmp/cofacet_early_exit_ab_clean.log \
                python benchmarks/ripser/harness/cofacet_early_exit_ab.py \
                --site $REPO/site_cpu_final --reps 3 \
                --out /tmp/cofacet_early_exit_ab_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) early-exit A/B done" >> "$LOG"

            # THE O3 PROFILE AND ITS THREAD SCALING. o3_4096 is the instance
            # we lose on, and the two tables say where the time goes and
            # whether the serial core is the problem.
            measure /tmp/o3_profile_clean.csv /tmp/o3_profile_clean.log \
                python benchmarks/ripser/harness/o3_profile.py \
                --out /tmp/o3_profile_clean.csv
            measure /tmp/o3_cpu_scaling_clean.csv /tmp/o3_cpu_scaling_clean.log \
                python benchmarks/ripser/harness/o3_cpu_scaling.py \
                --out /tmp/o3_cpu_scaling_clean.csv
            echo "$(date +%Y-%m-%dT%H:%M:%S) o3 profile and thread scaling done" >> "$LOG"
        else
            echo "$(date +%Y-%m-%dT%H:%M:%S) the scale scan was contaminated -- waiting for another window" >> "$LOG"
        fi
        exit 0
    fi
    sleep 60
done
echo "$(date +%Y-%m-%dT%H:%M:%S) gave up waiting for a quiet window" >> "$LOG"
exit 1
