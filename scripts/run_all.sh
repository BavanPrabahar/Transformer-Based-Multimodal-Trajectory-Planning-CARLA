#!/usr/bin/env bash
# End-to-end pipeline: CARLA server -> expert data -> features -> training -> closed-loop eval -> report.
# Usage: CARLA_ROOT=~/CARLA scripts/run_all.sh [episodes] [eval_per_template]
set -euo pipefail
cd "$(dirname "$0")/.."
EPISODES=${1:-600}
PER_TPL=${2:-10}
CARLA_ROOT=${CARLA_ROOT:-$HOME/CARLA}
PY=.venv/bin/python
mkdir -p logs

if ! $PY -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(3); c.get_server_version()" 2>/dev/null; then
  LOG_DIR=logs scripts/carla_server.sh "$CARLA_ROOT" 2000 &
  for i in $(seq 1 60); do
    $PY -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(3); c.get_server_version()" 2>/dev/null && break
    sleep 3
  done
fi

$PY -u scripts/calibrate_lowlevel.py --town Town01 | tee logs/calibration.log
$PY -u scripts/collect_data.py --towns Town01 Town03 Town05 --episodes "$EPISODES" --out data/train | tee logs/collect.log
$PY -u scripts/build_dataset.py --data data/train --out data/features --stride 2 | tee logs/build.log
$PY -u scripts/train.py --data data/features --epochs 40 --out checkpoints/planner.pt | tee logs/train.log
$PY -u scripts/evaluate.py --town Town02 --per-template "$PER_TPL" \
    --controllers lab_mpc capsule_nmpc expert hybrid hybrid_noselector hybrid_nosafety \
    --model checkpoints/planner.pt --out results/town02 | tee logs/eval.log
$PY scripts/report.py --results results/town02 --history checkpoints/planner_history.json --out results
