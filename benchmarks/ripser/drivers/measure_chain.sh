#!/bin/bash
# Run the two measurement chains, SEQUENTIALLY.
#
# WHY THEY MUST NOT RUN AT THE SAME TIME. Each driver waits for a quiet
# window on its own, and the quiet-window test counts the cores used by
# processes that are NOT ours. So OUR OWN chain does not make the box look
# busy, and two drivers started together would both start at once -- and
# then they would compete for the same sixteen cores and the same A100, and
# every number in both tables would be wrong in a way the gate cannot see.
#
# The gate protects against another USER. It says nothing about us.
#
# So the order is explicit: the classical chain first (it carries the
# head-to-head and the o3 row), then the corpus chain (the 582-cell table
# and the two corpus tables). Each driver still waits for its own quiet
# window before it starts, so a neighbour's job still stops the whole thing.
# NOT `set -u`: conda's activate script reads ADDR2LINE, which is not set in
# a non-interactive shell, and `set -u` turns that into a fatal error that
# kills this script before it measures anything. The same trap is in
# corpus_quiet_run.sh and build_measure_sites.sh.
set -eo pipefail

REPO=${CANNS_BENCH_REPO:-/home/sichaohe/canns-lib-dev}
LOG=/tmp/measure_chain.log

cd "$REPO" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl

echo "$(date +%Y-%m-%dT%H:%M:%S) chain 1/2: the classical chain" >> "$LOG"
bash benchmarks/ripser/drivers/quiet_window_watch.sh >> "$LOG" 2>&1
echo "$(date +%Y-%m-%dT%H:%M:%S) chain 1/2 done" >> "$LOG"

echo "$(date +%Y-%m-%dT%H:%M:%S) chain 2/2: the corpus chain" >> "$LOG"
bash benchmarks/ripser/drivers/corpus_quiet_run.sh >> "$LOG" 2>&1
echo "$(date +%Y-%m-%dT%H:%M:%S) chain 2/2 done" >> "$LOG"
