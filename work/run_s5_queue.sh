#!/bin/bash
# Sequential runner for the 5080 campaign: executes each case's job script
# in its run dir, one at a time (the whole machine is the resource), and
# appends a one-line verdict per case to s5_queue.progress.
# Usage from work/:  ./run_s5_queue.sh s5f_bnc64_o1 s5f_bnc64_g1 ...
set -u
cd "$(dirname "$0")" || exit 1
LOG=s5_queue.progress
for c in "$@"; do
  if [ ! -x "$c/$c.sh" ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] $c MISSING-SCRIPT" >> "$LOG"; continue
  fi
  t0=$SECONDS
  ( cd "$c" && bash "$c.sh" > "$c.joblog" 2>&1 )
  rc=$?
  wall=$((SECONDS-t0))
  tl=$(grep -hm1 "^TIMING ${c} " "$c/$c.joblog" 2>/dev/null || echo "TIMING $c NONE")
  th=$(grep -A2 "AFTER run" "$c/$c.env" 2>/dev/null | grep -cE "SW Thermal Slowdown.*Active|HW Thermal Slowdown.*Active")
  echo "[$(date '+%m-%d %H:%M:%S')] $c rc=$rc wall=${wall}s thermal_flags=$th | $tl" >> "$LOG"
done
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE-DONE ($# cases)" >> "$LOG"
