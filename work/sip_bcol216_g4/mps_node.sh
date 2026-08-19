#!/bin/bash
# Per-node MPS helper for sidia333_nc_2node.
#
# The batch script only ever executes on the first allocated node, but MPS is a
# per-node service: every node that hosts ranks needs its own control daemon and
# its own node-local pipe directory.  So this is launched once per node with
#   mpirun -np $NNODE -npernode 1 ./mps_node.sh <verb>
# and mpirun's -x carries CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY over.
#
# Kept as a separate file rather than an inline `bash -c '...'` so the quoting
# stays readable.

set -u

case "${1:-}" in

  start)
    # /tmp is node-local, so each node creates its own copy of these paths.
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" || exit 1
    nvidia-cuda-mps-control -d
    # Give the daemon a moment to publish the control pipe before "check" runs.
    for _ in $(seq 1 20); do
      [ -e "${CUDA_MPS_PIPE_DIRECTORY}/control" ] && break
      sleep 0.5
    done
    ;;

  check)
    # One line per node; the caller greps this for a "no" and aborts.
    printf '%s: control_pipe=%s daemons=%s gpu=%s\n' \
      "$(hostname)" \
      "$([ -e "${CUDA_MPS_PIPE_DIRECTORY}/control" ] && echo yes || echo no)" \
      "$(pgrep -fc nvidia-cuda-mps-control 2>/dev/null || echo 0)" \
      "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    ;;

  report)
    # Post-run: did an MPS *server* (not just the control daemon) ever appear?
    printf '%s: mps_server=%s gpu_mem_used=%s\n' \
      "$(hostname)" \
      "$(pgrep -fc nvidia-cuda-mps-server 2>/dev/null || echo 0)" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
    ;;

  stop)
    if [ -e "${CUDA_MPS_PIPE_DIRECTORY}/control" ]; then
      echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
    fi
    rm -rf "$(dirname "$CUDA_MPS_PIPE_DIRECTORY")"
    ;;

  *)
    echo "usage: $0 {start|check|report|stop}" >&2
    exit 2
    ;;
esac
