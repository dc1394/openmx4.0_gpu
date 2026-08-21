#!/bin/bash
# RTX 5080 production / preflight case generator (rtx5080_procedure.md,
# plan v2.6 sec. 7.2 / 8.2 / 8.3).  Local single-node port of gen_siprod.sh:
# no scheduler -- each case gets a directly-executable job script with the
# same conventions as the Pegasus template (MPS start/verify/stop, 5-s
# nvidia-smi sampler, watchdog, machine-readable TIMING line, .env record
# incl. CLOCK/POWER/PERFORMANCE snapshots before/after per procedure 0-6).
#
#   s5p_<s><mode><atoms>_<cfg><rep>     (production: 25 SCF, criterion 1e-15)
#   s5f_<s><mode><atoms>_<cfg><rep>     (preflight:   3 SCF, criterion 1e-13)
#     s    : b = band (Kgrid 2x2x2), c = cluster (Gamma-only)
#     mode : col | nc
#     cfg  : c16 = CPU elpa2, flat MPI 16 ranks (no GPU; reference)
#            o   = GPU gpusolver + MPS on,  gemmul8 off (cuBLAS FP64)
#            g   = GPU gpusolver + MPS on,  gemmul8 on  (INT8 emulation)
#            om  = GPU gpusolver + MPS OFF, gemmul8 off (plan 8.3 ablation)
#            gm  = GPU gpusolver + MPS OFF, gemmul8 on  (plan 8.3 interaction)
#
# All decks derive from the frozen probe decks sib_<mode><atoms>_p exactly as
# on Pegasus (System.Name; solver/Kgrid flip for s=c; eigen.lib per cfg;
# gemmul8 line kept only for o/om, dropped for g/gm/CPU (on = build default);
# production: scf.maxIter 3->25, criterion 1e-13->1e-15).
# Rules of record: -np 16 -nt 1 fixed for every 5080 series; wall time =
# Max_Time (slowest rank); 5 reps o/g, 3 reps c16/om/gm; a rep that shows
# GPU thermal slowdown in the post-run PERFORMANCE snapshot is discarded and
# re-measured under a new rep number (procedure 0-6).
#
# Usage from work/:  ./gen_s5p.sh [-p] b:col:216:o:11 c:nc:216:om:11 ...
set -u
cd "$(dirname "$0")" || exit 1

PRE=s5p; PREFLIGHT=0
if [ "${1:-}" = "-p" ]; then PRE=s5f; PREFLIGHT=1; shift; fi

NVROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
fail=0; ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

# per-case mpirun --timeout (s): generous; a timeout kill is itself a result.
timeout_for() {  # $1=s $2=mode $3=atoms $4=cfg $5=preflight
  local t
  if [ "$5" = 1 ]; then
    t=900
    [ "$3" -ge 384 ] && t=1800
    [ "$3" -ge 512 ] && t=2700
    [ "$4" = c16 ] && t=$((t*3))
  else
    case "$2$3" in
      nc64)  t=900;;  nc128) t=1500;; nc216) t=2700;;
      col216) t=1500;; col384) t=3600;; col512) t=5400;;
      *) t=3600;;
    esac
    [ "$1" = b ] && [ "$2" = nc ] && [ "$3" = 216 ] && t=3600
    case "$4" in
      om|gm) t=$((t*2));;
      c16)   t=$((t*4)); [ "$1" = b ] && [ "$2" = nc ] && t=14400;;
    esac
  fi
  echo "$t"
}

header() {  # $1=case $2=s $3=mode $4=atoms $5=cfg $6=scf
  local what solver="band solver, Kgrid 2x2x2 (8 computed k)"
  [ "$2" = c ] && solver="cluster solver, Gamma-only"
  case "$5" in
    c16) what="CPU elpa2, flat MPI 16 ranks, no GPU (reference per plan 8.2)";;
    o)   what="GPU gpusolver + MPS on, gemmul8 off -> cuBLAS FP64 (Layer 3 baseline)";;
    g)   what="GPU gpusolver + MPS on, gemmul8 on -> INT8 FP64 emulation (Layer 3)";;
    om)  what="GPU gpusolver, MPS OFF, gemmul8 off (plan 8.3 MPS ablation)";;
    gm)  what="GPU gpusolver, MPS OFF, gemmul8 on (plan 8.3 MPS x GEMMul8 interaction)";;
  esac
  cat <<EOF
# RTX 5080 local benchmark (rtx5080_procedure.md; plan v2.6 sec. 7.2/8.2/8.3): $1.
# $4 Si atoms from sidia.dat supercells, $solver.
# Config: $what.
# Fixed $6 SCF, MD.Type Opt / MD.maxIter 1; wall time = Max_Time column
# (slowest rank).  All 5080 series: -np 16 -nt 1 on Core i9-10980XE,
# 1x RTX 5080 16 GB (sm_120), binary = v2.0_thesis tag build.
# A crash here is a result, not a harness failure.
EOF
}

emit_gpu_script() {  # $1=case $2=s $3=tmo $4=mps(on|off) $5=g8txt
  local c=$1 s=$2 tmo=$3 mps=$4 g8=$5 rpt1 rpt2
  if [ "$s" = b ]; then
    rpt1='echo "=== Band GPU dispatch / fallback lines ==="'
    rpt2='grep -E "<Band_DFT_Col>|<Band_DFT_NonCol>|<Band>" "$STDOUT" 2>/dev/null | head -12'
  else
    rpt1='echo "=== Cluster GPU fallback (expect none) + Set_Hamiltonian GPU lines ==="'
    rpt2='grep -E "<Cluster_DFT_(Col|NonCol)>|<Set_Hamiltonian> GPU device" "$STDOUT" 2>/dev/null | head -8'
  fi
  local mpsstart mpsassert mpsnote
  if [ "$mps" = on ]; then
    mpsstart='./mps_node.sh start >> "$ENVLOG" 2>&1'
    mpsassert=yes
    mpsnote="MPS on"
  else
    mpsstart=': # MPS OFF: no daemon started (plan 8.3 ablation); unique pipe dir keeps any stray daemon unreachable'
    mpsassert=no
    mpsnote="MPS OFF"
  fi
  cat <<EOS
set -u
ulimit -c 0

NVROOT=$NVROOT
NPROC=16
CASE=$c
STDOUT=\${CASE}.std
STDERR=\${CASE}.err
ENVLOG=\${CASE}.env
SMILOG=\${CASE}.smi
PROGLOG=\${CASE}.progress

progress() { echo "[\$(date '+%H:%M:%S')] \$*" >> "\$PROGLOG"; }

cd "\$(dirname "\$0")" || exit 1
: > "\$PROGLOG"
: > "\$SMILOG"
progress "script start on \$(hostname)"

unset LD_LIBRARY_PATH
export PATH="\$NVROOT/comm_libs/mpi/bin:\$NVROOT/compilers/bin:\$PATH"

# NVIDIA MPS -- job-unique node-local pipe dir (short: unix-socket sun_path).
JOBTAG=\$\$
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps5-\${JOBTAG}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps5-\${JOBTAG}/log"
mkdir -p "\$CUDA_MPS_PIPE_DIRECTORY" "\$CUDA_MPS_LOG_DIRECTORY"
MPS_X="-x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY"
mps_stop() { ./mps_node.sh stop >/dev/null 2>&1; }
trap mps_stop EXIT INT TERM

{
  echo "=== host      : \$(hostname)"
  echo "=== date      : \$(date)"
  echo "=== layout    : 1 node x \${NPROC} ranks, 1 RTX 5080, $mpsnote, gemmul8 $g8"
  echo "=== eigen library / gemmul8 requested by the input ==="
  grep -E "scf.eigen.lib|scf.gemmul8" "\${CASE}.dat" || true
  echo "=== host memory before the run ==="
  free -g
  echo "=== GPU ==="
  nvidia-smi -L 2>&1 || true
  echo "=== GPU clocks/power/throttle BEFORE run (procedure 0-6) ==="
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>&1 | grep -E "Clocks Event|Graphics *:|SM *:|Memory *:|Power Draw|Current Temp|SW Thermal|HW Thermal|SW Power|HW Power|Active" | head -30
} > "\$ENVLOG" 2>&1

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "FATAL: nvidia-cuda-mps-control not found; aborting." | tee -a "\$ENVLOG"
  exit 1
fi

# the GPU must be exclusively ours (procedure sec. 2)
FOREIGN=\$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -cv '^\$')
if [ "\$FOREIGN" -ne 0 ]; then
  echo "FATAL: \$FOREIGN foreign GPU compute process(es) present; refusing to measure:" | tee -a "\$ENVLOG"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tee -a "\$ENVLOG"
  exit 1
fi

$mpsstart

{ echo "=== MPS ==="
  echo "pipe dir = \${CUDA_MPS_PIPE_DIRECTORY}"
  ./mps_node.sh check
} > mps_check.tmp 2>&1
cat mps_check.tmp >> "\$ENVLOG"

# assert the MPS state under test (a wrong state would poison the rep)
if [ "\$(grep -c 'control_pipe=$mpsassert' mps_check.tmp)" -ne 1 ]; then
  echo "FATAL: expected this node with control_pipe=$mpsassert, got:" | tee -a "\$ENVLOG"
  cat mps_check.tmp | tee -a "\$ENVLOG"
  rm -f mps_check.tmp
  exit 1
fi
rm -f mps_check.tmp
progress "MPS state verified (control_pipe=$mpsassert)"

( trap - EXIT INT TERM
  for _ in \$(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " \$3 "/" \$2}'
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    } >> "\$SMILOG" 2>&1
    sleep 5
  done ) &
SMIPID=\$!

progress "launching mpirun: \${NPROC} ranks"
mpirun -np "\${NPROC}" --timeout $tmo \\
       \${MPS_X} \\
       ../openmx "\${CASE}.dat" -nt 1 > "\$STDOUT" 2> "\$STDERR" &
MPIPID=\$!
progress "mpirun launched (pid \${MPIPID})"

DONE_MARK="The calculation was normally finished."
FAIL_RE="exited on signal|prterun noticed|Out Of Memory|oom-kill"
verdict=""
while kill -0 "\$MPIPID" 2>/dev/null; do
  if grep -qF "\$DONE_MARK" "\$STDOUT" 2>/dev/null; then
    verdict="done"; grace=60
  elif grep -qEs "\$FAIL_RE" "\$STDERR" "\$STDOUT"; then
    verdict="crash"; grace=120
  fi
  if [ -n "\$verdict" ]; then
    progress "\${verdict} detected; \${grace} s grace for teardown"
    sleep "\$grace"
    if kill -0 "\$MPIPID" 2>/dev/null; then
      progress "mpirun still up after grace -> SIGTERM"
      echo "watchdog: \${verdict} detected, mpirun still up after \${grace} s; terminating" >> "\$STDERR"
      kill -TERM "\$MPIPID" 2>/dev/null
      sleep 20
      kill -0 "\$MPIPID" 2>/dev/null && { progress "still up -> SIGKILL"; kill -KILL "\$MPIPID" 2>/dev/null; }
    else
      progress "mpirun exited on its own during grace"
    fi
    break
  fi
  sleep 5
done
progress "watchdog loop exited (verdict='\${verdict:-none}'); reaping mpirun"

wait "\$MPIPID"
rc=\$?
progress "mpirun reaped rc=\${rc}"

pkill -KILL -P "\$SMIPID" 2>/dev/null
kill -KILL "\$SMIPID" 2>/dev/null
progress "sampler killed"

{
  echo "=== GPU clocks/power/throttle AFTER run (procedure 0-6) ==="
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>&1 | grep -E "Clocks Event|Graphics *:|SM *:|Memory *:|Power Draw|Current Temp|SW Thermal|HW Thermal|SW Power|HW Power|Active" | head -30
} >> "\$ENVLOG" 2>&1

if [ "\$rc" -ne 0 ] && grep -qF "\$DONE_MARK" "\$STDOUT" 2>/dev/null; then
  echo "=== mpirun returned \${rc}, but openmx printed the completion banner;" >> "\$STDOUT"
  echo "=== treating as success (teardown was killed by the watchdog). ===" >> "\$STDOUT"
  rc=0
fi

echo "=== openmx exit status = \${rc} (\$(date)) ===" >> "\$STDOUT"

echo "=== TIMING (machine-readable; Max_Time column, seconds) ==="
awk -v c="\${CASE}" '
  /Computational Time \(second\)/ {inblk=1}
  inblk && /^ *Total Computational Time *=/ {tot=\$NF}
  inblk && /^ *DFT *=/                      {dft=\$NF}
  inblk && /^ *Diagonalization *=/          {diag=\$NF}
  inblk && /^ *Set_Hamiltonian *=/          {sh=\$NF}
  END {
    if (tot=="") { printf "TIMING %s INCOMPLETE\n", c }
    else { printf "TIMING %s total=%s dft=%s diag=%s set_hamiltonian=%s\n", c, tot, dft, diag, sh }
  }' "\$STDOUT"

echo "=== full timing block ==="
sed -n '/Computational Time (second)/,/^The calculation/p' "\$STDOUT" 2>/dev/null | head -35
echo "=== Utot ==="
grep -E "Utot\." "\${CASE}.out" 2>/dev/null | tail -2
echo "=== SCF iterations completed ==="
grep -cE "SCF=" "\$STDOUT" 2>/dev/null
$rpt1
$rpt2
echo "=== peak host memory, this node (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "\$SMILOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== peak GPU, this node ==="
grep -oE "^[0-9]+ %, [0-9]+ MiB" "\$SMILOG" 2>/dev/null | sort -t, -k2 -rn | head -3
echo "=== MPS server (post-run) ==="
./mps_node.sh report 2>&1
echo "=== teardown hang? (watchdog line present => yes) ==="
grep -c "watchdog:" "\$STDERR" 2>/dev/null
echo "=== progress markers ==="
cat "\$PROGLOG" 2>/dev/null

exit \${rc}
EOS
}

emit_cpu_script() {  # $1=case $2=s $3=tmo
  local c=$1 s=$2 tmo=$3
  cat <<EOS
set -u
ulimit -c 0

NVROOT=$NVROOT
NPROC=16
CASE=$c
STDOUT=\${CASE}.std
STDERR=\${CASE}.err
ENVLOG=\${CASE}.env
SMILOG=\${CASE}.smi
PROGLOG=\${CASE}.progress

progress() { echo "[\$(date '+%H:%M:%S')] \$*" >> "\$PROGLOG"; }

cd "\$(dirname "\$0")" || exit 1
: > "\$PROGLOG"
: > "\$SMILOG"
progress "script start on \$(hostname)"

unset LD_LIBRARY_PATH
export PATH="\$NVROOT/comm_libs/mpi/bin:\$NVROOT/compilers/bin:\$PATH"

{
  echo "=== host      : \$(hostname)"
  echo "=== date      : \$(date)"
  echo "=== layout    : 1 node x \${NPROC} ranks, CPU elpa2 (no GPU)"
  echo "=== eigen library / gemmul8 requested by the input ==="
  grep -E "scf.eigen.lib|scf.gemmul8" "\${CASE}.dat" || true
  echo "=== host memory before the run ==="
  free -g
} > "\$ENVLOG" 2>&1

( for _ in \$(seq 1 3000); do
    { date '+--- %H:%M:%S'
      free -g | awk '/^Mem:/ {print "host mem used/total (GiB): " \$3 "/" \$2}'
    } >> "\$SMILOG" 2>&1
    sleep 5
  done ) &
SMIPID=\$!

progress "launching mpirun: \${NPROC} ranks"
mpirun -np "\${NPROC}" --timeout $tmo \\
       ../openmx "\${CASE}.dat" -nt 1 > "\$STDOUT" 2> "\$STDERR" &
MPIPID=\$!
progress "mpirun launched (pid \${MPIPID})"

DONE_MARK="The calculation was normally finished."
FAIL_RE="exited on signal|prterun noticed|Out Of Memory|oom-kill"
verdict=""
while kill -0 "\$MPIPID" 2>/dev/null; do
  if grep -qF "\$DONE_MARK" "\$STDOUT" 2>/dev/null; then
    verdict="done"; grace=60
  elif grep -qEs "\$FAIL_RE" "\$STDERR" "\$STDOUT"; then
    verdict="crash"; grace=120
  fi
  if [ -n "\$verdict" ]; then
    progress "\${verdict} detected; \${grace} s grace for teardown"
    sleep "\$grace"
    kill -0 "\$MPIPID" 2>/dev/null && { kill -TERM "\$MPIPID" 2>/dev/null; sleep 20; }
    kill -0 "\$MPIPID" 2>/dev/null && kill -KILL "\$MPIPID" 2>/dev/null
    break
  fi
  sleep 5
done

wait "\$MPIPID"
rc=\$?
progress "mpirun reaped rc=\${rc}"
pkill -KILL -P "\$SMIPID" 2>/dev/null
kill -KILL "\$SMIPID" 2>/dev/null

if [ "\$rc" -ne 0 ] && grep -qF "\$DONE_MARK" "\$STDOUT" 2>/dev/null; then
  rc=0
fi
echo "=== openmx exit status = \${rc} (\$(date)) ===" >> "\$STDOUT"

echo "=== TIMING (machine-readable; Max_Time column, seconds) ==="
awk -v c="\${CASE}" '
  /Computational Time \(second\)/ {inblk=1}
  inblk && /^ *Total Computational Time *=/ {tot=\$NF}
  inblk && /^ *DFT *=/                      {dft=\$NF}
  inblk && /^ *Diagonalization *=/          {diag=\$NF}
  inblk && /^ *Set_Hamiltonian *=/          {sh=\$NF}
  END {
    if (tot=="") { printf "TIMING %s INCOMPLETE\n", c }
    else { printf "TIMING %s total=%s dft=%s diag=%s set_hamiltonian=%s\n", c, tot, dft, diag, sh }
  }' "\$STDOUT"

echo "=== full timing block ==="
sed -n '/Computational Time (second)/,/^The calculation/p' "\$STDOUT" 2>/dev/null | head -35
echo "=== Utot ==="
grep -E "Utot\." "\${CASE}.out" 2>/dev/null | tail -2
echo "=== SCF iterations completed ==="
grep -cE "SCF=" "\$STDOUT" 2>/dev/null
echo "=== peak host memory, this node (GiB) ==="
grep -oE "host mem used/total \(GiB\): [0-9]+/[0-9]+" "\$SMILOG" 2>/dev/null | sort -t: -k2 -rn | head -3
echo "=== progress markers ==="
cat "\$PROGLOG" 2>/dev/null

exit \${rc}
EOS
}

for pt in "$@"; do
  IFS=: read -r s mode atoms cfg rep <<<"$pt"
  src="sib_${mode}${atoms}_p"
  [ -s "$src/$src.dat" ] || { say "FAIL missing probe deck $src"; fail=1; continue; }
  case "$s" in b|c) : ;; *) say "FAIL bad solver tag $s"; fail=1; continue;; esac
  case "$cfg" in c16|o|g|om|gm) : ;; *) say "FAIL bad cfg $cfg"; fail=1; continue;; esac
  c="${PRE}_${s}${mode}${atoms}_${cfg}${rep}"
  ngen=$((ngen+1)); mkdir -p "$c"

  # ---- deck -----------------------------------------------------------
  sed -e "s/^System.Name  *${src}\$/System.Name                     ${c}/" \
      "$src/$src.dat" > "$c/$c.dat"
  if [ "$PREFLIGHT" = 0 ]; then
    sed -i -e "s/^\(scf.maxIter  *\)3 .*\$/\125          # fixed-iteration timing/" \
           -e "s/^scf.criterion  *1.0e-13.*\$/scf.criterion             1.0e-15      # pin the SCF count at 25/" \
           "$c/$c.dat"
  fi
  if [ "$s" = c ]; then
    sed -i -e "s/^\(scf.EigenvalueSolver  *\)Band .*\$/\1cluster     # DC|GDC|Cluster|Band/" \
           -e "s/^\(scf.Kgrid  *\)2 2 2.*\$/\11 1 1       # Gamma-only for the cluster solver/" "$c/$c.dat"
  fi
  case "$cfg" in
    c16)   sed -i -e "s/^\(scf.eigen.lib  *\)gpusolver\$/\1elpa2/" -e "/^scf.gemmul8.enable/d" "$c/$c.dat";;
    g|gm)  sed -i -e "/^scf.gemmul8.enable/d" "$c/$c.dat";;
    o|om)  : ;;  # keep the explicit gemmul8-off line from the probe deck
  esac

  # ---- job script ------------------------------------------------------
  tmo=$(timeout_for "$s" "$mode" "$atoms" "$cfg" "$PREFLIGHT")
  scfn=25; [ "$PREFLIGHT" = 1 ] && scfn=3
  {
    echo '#!/bin/bash'
    header "$c" "$s" "$mode" "$atoms" "$cfg" "$scfn"
    case "$cfg" in
      c16) emit_cpu_script "$c" "$s" "$tmo";;
      o|g) emit_gpu_script "$c" "$s" "$tmo" on  "$([ "$cfg" = g ] && echo on || echo off)";;
      om)  emit_gpu_script "$c" "$s" "$tmo" off off;;
      gm)  emit_gpu_script "$c" "$s" "$tmo" off on;;
    esac
  } > "$c/$c.sh"
  chmod +x "$c/$c.sh"
  case "$cfg" in o|g|om|gm) cp mps_node.sh "$c/mps_node.sh"; chmod +x "$c/mps_node.sh";; esac

  # ---- assertions ------------------------------------------------------
  say "== $c (timeout ${tmo}s)"
  need "System.Name"  "$c"       "$(awk '$1=="System.Name"{print $2}' "$c/$c.dat")"
  need "solver"       "$([ "$s" = c ] && echo cluster || echo Band)" "$(awk '$1=="scf.EigenvalueSolver"{print $2}' "$c/$c.dat")"
  need "Kgrid"        "$([ "$s" = c ] && echo '1 1 1' || echo '2 2 2')" "$(awk '$1=="scf.Kgrid"{print $2, $3, $4}' "$c/$c.dat")"
  need "spin"         "$([ "$mode" = nc ] && echo NC || echo off)" "$(awk '$1=="scf.SpinPolarization"{print $2}' "$c/$c.dat")"
  case "$cfg" in c16) explib=elpa2;; *) explib=gpusolver;; esac
  need "eigen.lib"    "$explib"  "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
  case "$cfg" in o|om) expg8=1;; *) expg8=0;; esac
  need "gemmul8 line" "$expg8"   "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")"
  if [ "$PREFLIGHT" = 1 ]; then
    need "scf.maxIter" "3"  "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
    need "criterion" "1.0e-13" "$(awk '$1=="scf.criterion"{print $2}' "$c/$c.dat")"
  else
    need "scf.maxIter" "25" "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
    need "criterion" "1.0e-15" "$(awk '$1=="scf.criterion"{print $2}' "$c/$c.dat")"
  fi
  need "atoms"        "$atoms"   "$(awk '$1=="Atoms.Number"{print $2}' "$c/$c.dat")"
  need "CASE var"     "CASE=${c}" "$(grep -m1 '^CASE=' "$c/$c.sh")"
  need "np16"         'mpirun -np "${NPROC}" --timeout '"${tmo}"' \' "$(grep -m1 '^mpirun -np' "$c/$c.sh")"
  case "$cfg" in
    o|g)   need "mps assert yes" "1" "$(grep -c "control_pipe=yes' mps_check.tmp" "$c/$c.sh")";;
    om|gm) need "mps assert no"  "1" "$(grep -c "control_pipe=no' mps_check.tmp" "$c/$c.sh")"
           need "no daemon start" "0" "$(grep -c '^\./mps_node.sh start' "$c/$c.sh")";;
  esac
  need "no stale refs" "0"      "$(grep -c "dia64dc\|DC-LNO\|${src}/\|PBS" "$c/$c.sh")"
  bash -n "$c/$c.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
done

if [ $fail -eq 0 ]; then say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"; else say "GENERATION HAD FAILURES"; exit 1; fi
