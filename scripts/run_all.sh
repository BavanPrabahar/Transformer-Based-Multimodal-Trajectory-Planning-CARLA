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
R="scripts/retry.sh 20"   # libcarla aborts on server time-outs; the scripts resume where they stopped
$R $PY -u scripts/collect_data.py --towns Town01 Town05 --episodes "$EPISODES" --out data/train | tee logs/collect.log
# first model on expert data
$PY -u scripts/build_dataset.py --data data/train --out data/features_v1 --stride 2 | tee logs/build_v1.log
$PY -u scripts/train.py --data data/features_v1 --epochs 30 --out checkpoints/planner_v1.pt | tee logs/train_v1.log
# one DAgger round: the learner drives (expert takes over when stuck), the expert labels
$R $PY -u scripts/collect_data.py --towns Town01 Town05 --episodes 150 --seed0 5000 --out data/dagger \
    --dagger checkpoints/planner_v1.pt | tee logs/collect_dagger.log
$PY -u scripts/build_dataset.py --data data/train data/dagger --out data/features --stride 2 | tee logs/build.log
$PY -u scripts/train.py --data data/features --epochs 40 --out checkpoints/planner.pt | tee logs/train.log
$PY -u scripts/evaluate.py --town Town02 --per-template "$PER_TPL" \
    --controllers lab_mpc capsule_nmpc expert hybrid hybrid_noselector hybrid_nosafety \
    --model checkpoints/planner.pt --out results/town02 | tee logs/eval.log
$PY scripts/report.py --results results/town02 --history checkpoints/planner_history.json --out results
