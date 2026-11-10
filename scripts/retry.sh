#!/usr/bin/env bash
# Re-run a resumable command until it exits 0 (libcarla aborts the process on server time-outs,
# which Python cannot catch). Usage: scripts/retry.sh <max_tries> <command...>
MAX=${1:-20}; shift
for i in $(seq 1 "$MAX"); do
  "$@" && exit 0
  echo "$(date '+%F %T') attempt $i exited with $? - retrying in 30 s" >&2
  sleep 30
done
exit 1
