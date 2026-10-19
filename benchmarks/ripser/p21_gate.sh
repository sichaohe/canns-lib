#!/usr/bin/env bash
# P21 gate: auto-sparse ON vs OFF, same binary, full 224-cell grid.
#
# Every fixture here is a point cloud turned into a distance matrix by cdist,
# so NO pair is missing and auto-sparse must never fire. That is exactly what
# this proves: the switch is inert on dense input. The POSITIVE case (a matrix
# that really is 30% missing) is the ASA matrix, checked bitwise in
# sparse_paths.py / p14_asa_gate.py.
set -u
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
PY=/home/sichaohe/miniconda3/envs/rl/bin/python
S=/home/sichaohe/canns-lib-dev/site_shard
export PYTHONPATH=$S
cd ~/ph_work

echo "### P21 GATE START $(date -Is)"
CANNS_RIPSER_NO_AUTO_SPARSE=1 $PY harness/apparent_bench.py \
  --mode equiv --out p21_off.csv 2>&1 | tail -2
CANNS_RIPSER_NO_AUTO_SPARSE=0 $PY harness/apparent_bench.py \
  --mode equiv --out p21_on.csv 2>&1 | tail -2
$PY compare_p4.py p21_off.csv p21_on.csv 0.02 2>&1 | tail -25
echo "### P21 GATE DONE $(date -Is)"
