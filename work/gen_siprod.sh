#!/bin/bash
# Layer-1 / Layer-3 / MPS-ablation production cases (research plan v2.6
# sec. 8.1 / 8.3, steps 12/14/15): one uniform case family for the H100
# 1-node benchmark matrix.
#
#   sip_<s><mode><atoms>_<cfg><rep>
#     s    : b = band (Kgrid 2x2x2), c = cluster (Gamma-only)
#     cfg  : c48/c36 = CPU elpa2, flat MPI 48/36 ranks (no GPU)
#            o  = GPU gpusolver + MPS on, gemmul8 off (cuBLAS FP64)
#            g  = GPU gpusolver + MPS on, gemmul8 on  (INT8 emulation)
#            om = GPU gpusolver, MPS OFF, gemmul8 off (ablation, plan 8.3)
#
# All decks derive from the frozen band probe decks sib_<mode><atoms>_p:
#   System.Name; solver/Kgrid for s=c; scf.eigen.lib per cfg; gemmul8 line
#   kept only for o/om (off) and dropped for g (on=default) and CPU;
#   scf.maxIter 25; scf.criterion 1e-13 -> 1e-15 (pin the SCF count; the
#   search showed 1e-13 can converge at 24).
# Job scripts: GPU cfgs from dia64dc_1n_g1 (MPS template; om variant flips
# the MPS assertion so the run REFUSES to proceed if a control pipe IS
# present, and never starts a daemon), CPU cfgs from dia64dc_1n_c1.
#
# Rules of record: 3 reps for c48/c36/om, 5 reps for o/g (plan 8.1);
# reps landing on bnode013/bnode033 are discarded and re-measured under a
# new rep number.  Wall time = Max_Time (slowest rank).
#
# Usage from work/:  ./gen_siprod.sh b:col:216:o:1 c:nc:216:om:2 ...
set -u
cd "$(dirname "$0")" || exit 1

GTPL=dia64dc_1n_g1
CTPL=dia64dc_1n_c1
[ -s "$GTPL/$GTPL.sh" ] && [ -s "$CTPL/$CTPL.sh" ] || { echo "FATAL: templates missing"; exit 1; }

fail=0; ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

header() {  # $1=case $2=s $3=mode $4=atoms $5=cfg
  local what solver="band solver, Kgrid 2x2x2 (8 computed k)"
  [ "$2" = c ] && solver="cluster solver, Gamma-only"
  case "$5" in
    c48) what="CPU elpa2, flat MPI 48 ranks, no GPU (Layer-1 CPU best-config)";;
    c36) what="CPU elpa2, flat MPI 36 ranks, no GPU (Layer-1 CPU alternate)";;
    o)   what="GPU gpusolver + MPS on, gemmul8 off -> cuBLAS FP64 (Layer 1/3 baseline)";;
    g)   what="GPU gpusolver + MPS on, gemmul8 on -> INT8 FP64 emulation (Layer 3)";;
    om)  what="GPU gpusolver, MPS OFF, gemmul8 off (plan 8.3 MPS ablation)";;
  esac
  cat <<EOF
# H100 1-node production benchmark (plan v2.6 sec. 8.1/8.3): $1.
# $4 Si atoms from sidia.dat supercells, $solver.
# Config: $what.
# Fixed 25 SCF (scf.criterion 1e-15), MD.Type Opt / MD.maxIter 1; wall
# time = Max_Time column (slowest rank).  Reps: 3 for CPU and MPS-off,
# 5 for the GPU o/g configs; slow-node draws (bnode013/bnode033) are
# discarded and re-measured under a new rep number.
# Binary: openmx cd5f0d5 + GEMMul8 v3.2.0 (md5 962f8d2519c2e6aa5a6295513f76fee9).
# A crash here is a result, not a harness failure.
EOF
}

rewrite_header() {
  local f=$1; shift
  local hdr; hdr=$(header "$@")
  awk -v hdr="$hdr" '
    /^#PBS -o /        { print; print ""; print hdr; skip=1; next }
    skip && /^set -u$/ { skip=0; print "" }
    skip               { next }
    { print }
  ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}

for pt in "$@"; do
  IFS=: read -r s mode atoms cfg rep <<<"$pt"
  src="sib_${mode}${atoms}_p"
  [ -s "$src/$src.dat" ] || { say "FAIL missing band deck $src"; fail=1; continue; }
  case "$s" in b|c) : ;; *) say "FAIL bad solver tag $s"; fail=1; continue;; esac
  case "$cfg" in c48|c36|o|g|om) : ;; *) say "FAIL bad cfg $cfg"; fail=1; continue;; esac
  c="sip_${s}${mode}${atoms}_${cfg}${rep}"
  jn="p${s}${mode:0:1}${atoms}${cfg}${rep}"
  ngen=$((ngen+1)); mkdir -p "$c"

  # ---- deck -----------------------------------------------------------
  sed -e "s/^System.Name  *${src}\$/System.Name                     ${c}/" \
      -e "s/^\(scf.maxIter  *\)3 .*\$/\125          # fixed-iteration timing/" \
      -e "s/^scf.criterion  *1.0e-13.*\$/scf.criterion             1.0e-15      # pin the SCF count at 25/" \
      "$src/$src.dat" > "$c/$c.dat"
  if [ "$s" = c ]; then
    sed -i -e "s/^\(scf.EigenvalueSolver  *\)Band .*\$/\1cluster     # DC|GDC|Cluster|Band/" \
           -e "s/^\(scf.Kgrid  *\)2 2 2.*\$/\11 1 1       # Gamma-only for the cluster solver/" "$c/$c.dat"
  fi
  case "$cfg" in
    c48|c36) sed -i -e "s/^\(scf.eigen.lib  *\)gpusolver\$/\1elpa2/" -e "/^scf.gemmul8.enable/d" "$c/$c.dat";;
    g)       sed -i -e "/^scf.gemmul8.enable/d" "$c/$c.dat";;
    o|om)    : ;;  # keep the explicit gemmul8-off line from the probe deck
  esac

  # ---- job script ------------------------------------------------------
  # walltimes: measured 25-SCF GPU refs are col300 132-150 s / nc216
  # 255-257 s; CPU is the open question -> give CPU band nc a wide slot.
  case "$cfg" in
    # CPU 3-SCF probes measured: bcol300 122 s, bnc216 164 s, cnc216 91 s
    # -> 25-SCF upper estimates ~15/20/8 min; 1.5x-4x headroom below.
    c48|c36) el=01:00:00; [ "$s" = b ] && [ "$mode" = nc ] && el=01:20:00;;
    *)       el=00:40:00; [ "$mode" = nc ] && el=01:00:00;;
  esac
  tmo=$(( $( IFS=:; set -- $el; echo $((10#$1*3600+10#$2*60+10#$3)) ) - 360 ))

  case "$cfg" in
    c48|c36)
      ranks=48; [ "$cfg" = c36 ] && ranks=36
      sed -e "s/${CTPL}/${c}/g" \
          -e "s/^PPN=48\$/PPN=${ranks}/" \
          -e "s/^#PBS -N .*\$/#PBS -N ${jn}/" \
          -e "s/^#PBS -l elapstim_req=.*\$/#PBS -l elapstim_req=${el}/" \
          -e "s/--timeout [0-9]*/--timeout ${tmo}/" \
          -e "s/of 50 max/of 25 max/" \
          "${CTPL}/${CTPL}.sh" > "$c/$c.sh"
      ;;
    o|g|om)
      if [ "$s" = b ]; then
        rpt1="s/^echo \"=== DC-LNO GPU dispatch lines ===\"\$/echo \"=== Band GPU dispatch \/ fallback lines ===\"/"
        rpt2="s|^grep -E \"<DC-LNO>\" .*\$|grep -E \"<Band_DFT_Col>\|<Band_DFT_NonCol>\|<Band>\" \"\$STDOUT\" 2>/dev/null \| head -12|"
      else
        rpt1="s/^echo \"=== DC-LNO GPU dispatch lines ===\"\$/echo \"=== Cluster GPU fallback (expect none) + Set_Hamiltonian GPU lines ===\"/"
        rpt2="s|^grep -E \"<DC-LNO>\" .*\$|grep -E \"<Cluster_DFT_(Col\|NonCol)>\|<Set_Hamiltonian> GPU device\" \"\$STDOUT\" 2>/dev/null \| head -8|"
      fi
      g8txt=off; [ "$cfg" = g ] && g8txt=on
      sed -e "s/${GTPL}/${c}/g" \
          -e "s/^#PBS -N .*\$/#PBS -N ${jn}/" \
          -e "s/^#PBS -l elapstim_req=.*\$/#PBS -l elapstim_req=${el}/" \
          -e "s/--timeout 1800/--timeout ${tmo}/" \
          -e "s/MPS on, gemmul8 on/MPS $([ "$cfg" = om ] && echo OFF || echo on), gemmul8 ${g8txt}/" \
          -e "s/of 50 max/of 25 max/" \
          -e "$rpt1" -e "$rpt2" \
          "${GTPL}/${GTPL}.sh" > "$c/$c.sh"
      if [ "$cfg" = om ]; then
        # MPS ablation: never start a daemon, and refuse to run if one is
        # unexpectedly present (mirror image of the MPS-on assertion).
        sed -i \
          -e "s/^mps_on_each_node start >> \"\$ENVLOG\" 2>&1\$/: # MPS OFF: no daemon started (plan 8.3 ablation)/" \
          -e "s/control_pipe=yes/control_pipe=no/" \
          -e "s/FATAL: expected \${NNODE} nodes with MPS/FATAL: expected \${NNODE} nodes WITHOUT MPS (ablation)/" \
          "$c/$c.sh"
      fi
      cp "${GTPL}/mps_node.sh" "$c/mps_node.sh"
      ;;
  esac
  rewrite_header "$c/$c.sh" "$c" "$s" "$mode" "$atoms" "$cfg"

  # ---- assertions ------------------------------------------------------
  say "== $c ($el)"
  need "System.Name"  "$c"       "$(awk '$1=="System.Name"{print $2}' "$c/$c.dat")"
  need "solver"       "$([ "$s" = c ] && echo cluster || echo Band)" "$(awk '$1=="scf.EigenvalueSolver"{print $2}' "$c/$c.dat")"
  need "Kgrid"        "$([ "$s" = c ] && echo '1 1 1' || echo '2 2 2')" "$(awk '$1=="scf.Kgrid"{print $2, $3, $4}' "$c/$c.dat")"
  need "spin"         "$([ "$mode" = nc ] && echo NC || echo off)" "$(awk '$1=="scf.SpinPolarization"{print $2}' "$c/$c.dat")"
  case "$cfg" in c48|c36) explib=elpa2;; *) explib=gpusolver;; esac
  need "eigen.lib"    "$explib"  "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
  case "$cfg" in o|om) expg8=1;; *) expg8=0;; esac
  need "gemmul8 line" "$expg8"   "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")"
  need "criterion"    "1.0e-15"  "$(awk '$1=="scf.criterion"{print $2}' "$c/$c.dat")"
  need "scf.maxIter"  "25"       "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
  need "atoms"        "$atoms"   "$(awk '$1=="Atoms.Number"{print $2}' "$c/$c.dat")"
  need "CASE var"     "CASE=${c}" "$(grep -m1 '^CASE=' "$c/$c.sh")"
  case "$cfg" in
    c48) need "PPN" "PPN=48" "$(grep -m1 '^PPN=' "$c/$c.sh")"; need "no gpunum" "0" "$(grep -c '^#PBS --gpunum' "$c/$c.sh")";;
    c36) need "PPN" "PPN=36" "$(grep -m1 '^PPN=' "$c/$c.sh")"; need "no gpunum" "0" "$(grep -c '^#PBS --gpunum' "$c/$c.sh")";;
    *)   need "PPN" "PPN=48" "$(grep -m1 '^PPN=' "$c/$c.sh")"; need "gpunum" "1" "$(grep -c '^#PBS --gpunum-lhost=1' "$c/$c.sh")";;
  esac
  if [ "$cfg" = om ]; then
    need "no daemon start" "0" "$(grep -c '^mps_on_each_node start' "$c/$c.sh")"
    need "inverted check"  "1" "$(grep -c 'control_pipe=no' "$c/$c.sh")"
  fi
  need "cpunum full"  "#PBS --cpunum-lhost=48" "$(grep -m1 '^#PBS --cpunum' "$c/$c.sh")"
  need "elapstim"     "#PBS -l elapstim_req=${el}" "$(grep -m1 '^#PBS -l elapstim' "$c/$c.sh")"
  need "no stale refs" "0"      "$(grep -c "dia64dc\|DC-LNO\|${src}/" "$c/$c.sh")"
  bash -n "$c/$c.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
done

if [ $fail -eq 0 ]; then say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"; else say "GENERATION HAD FAILURES"; exit 1; fi
