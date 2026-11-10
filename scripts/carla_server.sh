#!/usr/bin/env bash
# Keep a headless CARLA 0.9.13 server alive (it occasionally segfaults on map loads).
# Usage: scripts/carla_server.sh [CARLA_ROOT] [PORT]
CARLA_ROOT=${1:-$HOME/CARLA}
PORT=${2:-2000}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
trap 'pkill -f "CarlaUE4-Linux-Shipping.*-carla-rpc-port=${PORT}"; exit 0' INT TERM
while true; do
  echo "$(date '+%F %T') starting CARLA on port ${PORT}" >> "$LOG_DIR/carla_supervisor.log"
  "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port="${PORT}" \
      >> "$LOG_DIR/carla_server.log" 2>&1
  echo "$(date '+%F %T') CARLA exited with code $? - restarting in 5 s" >> "$LOG_DIR/carla_supervisor.log"
  sleep 5
done
