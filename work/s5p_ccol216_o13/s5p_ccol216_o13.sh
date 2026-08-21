#!/bin/bash
# RTX 5080 local benchmark (rtx5080_procedure.md; plan v2.6 sec. 7.2/8.2/8.3): s5p_ccol216_o13.
# 216 Si atoms from sidia.dat supercells, cluster solver, Gamma-only.
# Config: GPU gpusolver + MPS on, gemmul8 off -> cuBLAS FP64 (Layer 3 baseline).
# Fixed 25 SCF, MD.Type Opt / MD.maxIter 1; wall time = Max_Time column
# (slowest rank).  All 5080 series: -np 16 -nt 1 on Core i9-10980XE,
# 1x RTX 5080 16 GB (sm_120), binary = v2.0_thesis tag build.
# A crash here is a result, not a harness failure.
set -u
ulimit -c 0

NVROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
NPROC=16
CASE=s5p_ccol216_o13
STDOUT=${CASE}.std
STDERR=${CASE}.err
ENVLOG=${CASE}.env
SMILOG=${CASE}.smi
PROGLOG=${CASE}.progress

progress() { echo "[$(date '+%H:%M:%S')] $*" >> "$PROGLOG"; }

cd "$(dirname "$0")" || exit 1
: > "$PROGLOG"
: > "$SMILOG"
progress "script start on $(hostname)"

unset LD_LIBRARY_PATH
export PATH="$NVROOT/comm_libs/mpi/bin:$NVROOT/compilers/bin:$PATH"

# NVIDIA MPS -- job-unique node-local pipe dir (short: unix-socket sun_path).
JOBTAG=$$
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps5-${JOBTAG}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps5-${JOBTAG}/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS_X="-x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY"
mps_stop() { ./mps_node.sh stop >/dev/null 2>&1; }
trap mps_stop EXIT INT TERM

{
  echo "=== host      : $(hostname)"
  echo "=== date      : $(date)"
  echo "=== layout    : 1 node x ${NPROC} ranks, 1 RTX 5080, MPS on, gemmul8 off"
  echo "=== eigen library / gemmul8 requested by the input ==="
  grep -E "scf.eigen.lib|scf.gemmul8" "${CASE}.dat" || true
  echo "=== host memory before the run ==="
  free -g
  echo "=== GPU ==="
  nvidia-smi -L 2>&1 || true
  echo "=== GPU clocks/power/throttle BEFORE run (procedure 0-6) ==="
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>&1 | grep -E "Clocks Event|Graphics *:|SM *:|Memory *:|Power Draw|Current Temp|SW Thermal|HW Thermal|SW Power|HW Power|Active" | head -30
} > "$ENVLOG" 2>&1

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "FATAL: nvidia-cuda-mps-control not found; aborting." | tee -a "$ENVLOG"
  exit 1
fi

# the GPU must be exclusively ours (procedure sec. 2)
FOREIGN=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -cv '^$')
if [ "$FOREIGN" -ne 0 ]; then
  echo "FATAL: $FOREIGN foreign GPU compute process(es) present; refusing to measure:" | tee -a "$ENVLOG"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tee -a "$ENVLOG"
  exit 1
fi

./mps_node.sh start >> "$ENVLOG" 2>&1

{ echo "=== MPS ==="
  echo "pipe dir = ${CUDA_MPS_PIPE_DIRECTORY}"
  ./mps_node.sh check
} > mps_check.tmp 2>&1
cat mps_check.tmp >> "$ENVLOG"

# assert the MPS state under test (a wrong state would poison the rep)
if [ "$(grep -c 'control_pipe=yes' mps_check.tmp)" -ne 1 ]; then
  echo "FATAL: expected this node with control_pipe=yes, got:" | tee -a "$ENVLOG"
  cat mps_check.tmp | tee -a "$ENVLOG"
  rm -f mps_check.tmp
  exit 1
fi
rm -f mps_check.tmp
progress "MPS state verified (control_pipe=yes)"

( trap - EXIT INT TERM
  for _ in $(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " $3 "/" $2}'
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    } >> "$SMILOG" 2>&1
    sleep 5
  done ) &
SMIPID=$!

progress "launching mpirun: ${NPROC} ranks"
mpirun -np "${NPROC}" --timeout 1500 \
       ${MPS_X} \
       ../openmx "${CASE}.dat" -nt 1 > "$STDOUT" 2> "$STDERR" &
MPIPID=$!
progress "mpirun launched (pid ${MPIPID})"

DONE_MARK="The calculation was normally finished."
FAIL_RE="exited on signal|prterun noticed|Out Of Memory|oom-kill"
verdict=""
while kill -0 "$MPIPID" 2>/dev/null; do
  if grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
    verdict="done"; grace=60
  elif grep -qEs "$FAIL_RE" "$STDERR" "$STDOUT"; then
    verdict="crash"; grace=120
  fi
  if [ -n "$verdict" ]; then
    progress "${verdict} detected; ${grace} s grace for teardown"
    sleep "$grace"
    if kill -0 "$MPIPID" 2>/dev/null; then
      progress "mpirun still up after grace -> SIGTERM"
      echo "watchdog: ${verdict} detected, mpirun still up after ${grace} s; terminating" >> "$STDERR"
      kill -TERM "$MPIPID" 2>/dev/null
      sleep 20
      kill -0 "$MPIPID" 2>/dev/null && { progress "still up -> SIGKILL"; kill -KILL "$MPIPID" 2>/dev/null; }
    else
      progress "mpirun exited on its own during grace"
    fi
    break
  fi
  sleep 5
done
progress "watchdog loop exited (verdict='${verdict:-none}'); reaping mpirun"

wait "$MPIPID"
rc=$?
progress "mpirun reaped rc=${rc}"

pkill -KILL -P "$SMIPID" 2>/dev/null
kill -KILL "$SMIPID" 2>/dev/null
progress "sampler killed"

{
  echo "=== GPU clocks/power/throttle AFTER run (procedure 0-6) ==="
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>&1 | grep -E "Clocks Event|Graphics *:|SM *:|Memory *:|Power Draw|Current Temp|SW Thermal|HW Thermal|SW Power|HW Power|Active" | head -30
} >> "$ENVLOG" 2>&1

if [ "$rc" -ne 0 ] && grep -qF "$DONE_MARK" "$STDOUT" 2>/dev/null; then
  echo "=== mpirun returned ${rc}, but openmx printed the completion banner;" >> "$STDOUT"
  echo "=== treating as success (teardown was killed by the watchdog). ===" >> "$STDOUT"
  rc=0
fi

echo "=== openmx exit status = ${rc} ($(date)) ===" >> "$STDOUT"

echo "=== TIMING (machine-readable; Max_Time column, seconds) ==="
awk -v c="${CASE}" '
  /Computational Time \(second\)/ {inblk=1}
  inblk && /^ *Total Computational Time *=/ {tot=$NF}
  inblk && /^ *DFT *=/                      {dft=$NF}
  inblk && /^ *Diagonalization *=/          {diag=$NF}
  inblk && /^ *Set_Hamiltonian *=/          {sh=$NF}
  END {
    if (tot=="") { printf "TIMING %s INCOMPLETE\n", c }
    else { printf "TIMING %s total=%s dft=%s diag=%s set_hamiltonian=%s\n", c, tot, dft, diag, sh }
  }' "$STDOUT"

echo "=== full timing block ==="
sed -n '/Computational Time (second)/,/^The calculation/p' "$STDOUT" 2>/dev/null | head -35
echo "=== Utot ==="
grep -E "Utot\." "${CASE}.out" 2>/dev/null | tail -2
echo "=== SCF iterations completed ==="
grep -cE "SCF=" "$STDOUT" 2>/dev/null
echo "=== Cluster GPU fallback (expect none) + Set_Hamiltonian GPU lines ==="
grep -E "<Cluster_DFT_(Col|NonCol)>|<Set_Hamiltonian> GPU device" "$STDOUT" 2>/dev/null | head -8
echo "=== peak host memory, this node (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "$SMILOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== peak GPU, this node ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "$SMILOG" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== MPS server (post-run) ==="
./mps_node.sh report 2>&1
echo "=== teardown hang? (watchdog line present => yes) ==="
grep -c "watchdog:" "$STDERR" 2>/dev/null
echo "=== progress markers ==="
cat "$PROGLOG" 2>/dev/null

exit ${rc}
