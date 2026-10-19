#!/bin/bash
# Build the shipping candidate from the current local tree and run the
# corrected head-to-head against ripser++.
#
# FEATURE SET: `gpu` MUST be listed. It is in neither `default` nor
# `[tool.maturin] features` (the released wheel has to build without nvcc),
# and a build without it silently falls back to the CPU apparent pre-pass and
# is ~35% slower -- which reads exactly like a regression in the change under
# test.
set -e
cd ~/canns-lib-dev
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl
export CARGO_TARGET_DIR=$HOME/canns-lib-dev/target

maturin build --release --out /tmp/wh_ship \
  --features parallel,hugepage,extension-module,gpu 2>&1 | tail -2
rm -rf ~/canns-lib-dev/site_ship
mkdir -p ~/canns-lib-dev/site_ship
cd /tmp/wh_ship && for w in *.whl; do unzip -o -q "$w" -d ~/canns-lib-dev/site_ship; done
cd ~/canns-lib-dev
echo "BUILT site_ship"

python benchmarks/ripser/clean_h2h.py \
  --out /home/sichaohe/ph_work/clean_h2h.csv --reps 3
echo "H2H DONE"
