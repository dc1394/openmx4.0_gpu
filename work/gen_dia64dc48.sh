#!/bin/bash
# Generate the dia64_dc-lno 1-node benchmark at -np 48:
#   dia64dc_1n48_{g,o,c}{1,2,3}
#   g = gpusolver + MPS, scf.gemmul8.enable on (default)
#   o = gpusolver + MPS, scf.gemmul8.enable off (plain cuBLAS FP64)
#   c = elpa2, CPU only, no GPU reserved
# 9 cases, 3 reps each.  The .dat comes from work/dia64_dc-lno/dia64_dc-lno.dat
# (64 C atoms, C6.0-s2p2d1 -> 832 basis functions, scf.EigenvalueSolver dc-lno,
# Gamma-only, 50 SCF max to 1e-10, orderN.HoppingRanges 6.0).
#
# Job scripts are derived from the s443nc 1-node cases, which are proven on
# this machine, with PPN 36 -> 48.  Note what the DC-LNO GPU path needs:
# Divide_Conquer_LNO.c dispatches a cluster to the GPU only when that
# cluster's local matrix dimension NUM >= 800 (OPENMX_DCLNO_GPU_THRESHOLD,
# default GPU_CPU_SWITCH_NUM2 = 800).  Whether this system clears that bar is
# a measurement, not an assumption -- the per-run .smi sampler records GPU
# utilisation and memory, so a GPU that never engages will show up there.
#
# Run from work/:  ./gen_dia64dc48.sh
set -u
cd "$(dirname "$0")" || exit 1

SRC=dia64_dc-lno/dia64_dc-lno.dat
[ -s "$SRC" ] || { echo "FATAL: $SRC missing"; exit 1; }

fail=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

header() {  # $1 = cfg
  local what
  case "$1" in
    g) what="GPU path (gpusolver + MPS, gemmul8 on)";;
    o) what="GPU path (gpusolver + MPS, gemmul8 OFF -> plain cuBLAS FP64)";;
    c) what="CPU path (elpa2, no GPU reserved)";;
  esac
  cat <<EOF
# dia64_dc-lno 1-node benchmark, ${what}.
# One of 9 requests: 1 node x 48 ranks x {gemmul8-on, gemmul8-off, elpa2} x 3
# reps, so Total and Diagonalization come with a mean.
# Binary: gpusolver2 merge (commit f5fb5b3).  64 C atoms, C6.0-s2p2d1 -> 832
# basis functions, scf.EigenvalueSolver dc-lno, Gamma-only, 50 SCF max to
# 1e-10, orderN.HoppingRanges 6.0 Ang in a 7.134 Ang cubic cell.
# DC-LNO sends a cluster to the GPU only when its local matrix dimension is
# >= 800 (Divide_Conquer_LNO.c, OPENMX_DCLNO_GPU_THRESHOLD).  The .smi
# sampler below is the evidence for whether that happened.
# 64 atoms over 48 ranks does not divide evenly: 16 ranks get 2 clusters and
# 32 get 1, so some load imbalance is expected at this rank count.
# A crash here is a result, not a failure of the harness.
EOF
}

rewrite_header() {  # $1 = file, $2 = cfg
  local f=$1 hdr
  hdr=$(header "$2")
  awk -v hdr="$hdr" '
    /^#PBS -o /        { print; print ""; print hdr; skip=1; next }
    skip && /^set -u$/ { skip=0; print "" }
    skip               { next }
    { print }
  ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}

for cfg in g o c; do
  tpl=s443nc_1n_${cfg}1
  case "$cfg" in
    g|o) lib=gpusolver; gpu=1;;
    c)   lib=elpa2;     gpu=0;;
  esac

  for r in 1 2 3; do
    c=dia64dc_1n48_${cfg}${r}
    mkdir -p "$c"

    sed -e "s/${tpl}/${c}/g" \
        -e "s/^PPN=36\$/PPN=48/" \
        -e "s/^#PBS -N .*\$/#PBS -N d64_48${cfg}${r}/" \
        -e "s/^echo \"=== Set_Hamiltonian GPU \/ residency lines ===\"\$/echo \"=== DC-LNO GPU dispatch lines ===\"/" \
        -e "s|^grep -E \"<Set_Hamiltonian> GPU (device\|resident)\" .*\$|grep -E \"<DC-LNO>\" \"\$STDOUT\" 2>/dev/null \| head -6|" \
        -e "s/of 25 max/of 50 max/" \
        -e "s/^echo \"=== Utot .*\$/echo \"=== Utot (no prior reference for this system) ===\"/" \
        "${tpl}/${tpl}.sh" > "${c}/${c}.sh"
    rewrite_header "${c}/${c}.sh" "$cfg"

    # the input deck itself comes from the user's dia64_dc-lno, not from s443nc
    sed -e "s/^\(System.Name  *\)dia64_dc-lno\$/\1${c}/" "$SRC" > "${c}/${c}.dat"
    if [ "$cfg" = c ]; then
      sed -i -e "s/^\(scf.eigen.lib  *\)gpusolver\$/\1elpa2/" "${c}/${c}.dat"
    elif [ "$cfg" = o ]; then
      sed -i -e "s/^\(scf.eigen.lib  *gpusolver\)\$/\1\nscf.gemmul8.enable        off       # A\/B: plain cuBLAS FP64/" "${c}/${c}.dat"
    fi

    if [ "$cfg" = c ]; then rm -f "${c}/mps_node.sh"; else cp "${tpl}/mps_node.sh" "${c}/mps_node.sh"; fi

    say "== ${c}"
    need "System.Name"    "$c"    "$(awk '$1=="System.Name"{print $2}' "${c}/${c}.dat")"
    need "solver"     "dc-lno"    "$(awk '$1=="scf.EigenvalueSolver"{print $2}' "${c}/${c}.dat")"
    need "scf.eigen.lib"  "$lib"  "$(awk '$1=="scf.eigen.lib"{print $2}' "${c}/${c}.dat")"
    need "gemmul8 line"   "$([ "$cfg" = o ] && echo 1 || echo 0)" \
                          "$(grep -c '^scf.gemmul8.enable' "${c}/${c}.dat")"
    need "gemmul8 value"  "$([ "$cfg" = o ] && echo off || echo -)" \
                          "$(awk '$1=="scf.gemmul8.enable"{print $2}' "${c}/${c}.dat" | grep . || echo -)"
    need "atoms"          "64"    "$(awk '$1=="Atoms.Number"{print $2}' "${c}/${c}.dat")"
    need "DATA.PATH"      "../../DFT_DATA19" "$(awk '$1=="DATA.PATH"{print $2}' "${c}/${c}.dat")"
    need "CASE var"   "CASE=${c}" "$(grep -m1 '^CASE=' "${c}/${c}.sh")"
    need "NNODE"      "NNODE=1"   "$(grep -m1 '^NNODE=' "${c}/${c}.sh")"
    need "PPN"        "PPN=48"    "$(grep -m1 '^PPN=' "${c}/${c}.sh")"
    need "-b nodes"   "#PBS -b 1" "$(grep -m1 '^#PBS -b ' "${c}/${c}.sh")"
    need "job name"   "#PBS -N d64_48${cfg}${r}" "$(grep -m1 '^#PBS -N ' "${c}/${c}.sh")"
    need "joblog"     "#PBS -o ${c}.joblog" "$(grep -m1 '^#PBS -o ' "${c}/${c}.sh")"
    need "cpunum-lhost" "#PBS --cpunum-lhost=48" "$(grep -m1 '^#PBS --cpunum' "${c}/${c}.sh")"
    need "gpunum-lhost" "$gpu" "$(grep -c '^#PBS --gpunum-lhost=1' "${c}/${c}.sh")"
    need "no stale refs" "0" "$(grep -c "${tpl}\|s443nc" "${c}/${c}.sh" "${c}/${c}.dat" | awk -F: '{s+=$2} END{print s}')"
    need "no PPN=36 left" "0" "$(grep -cE 'PPN=36|npernode 36' "${c}/${c}.sh")"
    # GPU cases must report the <DC-LNO> dispatch lines; the CPU case has no
    # such section (its template never had the Set_Hamiltonian GPU block).
    need "DC-LNO report line" "$([ "$cfg" = c ] && echo 0 || echo 1)" \
                              "$(grep -cF 'grep -E "<DC-LNO>"' "${c}/${c}.sh")"
    need "mps_node.sh"   "$([ "$cfg" = c ] && echo no || echo yes)" \
                         "$([ -s "${c}/mps_node.sh" ] && echo yes || echo no)"
    bash -n "${c}/${c}.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
  done
done

if [ $fail -eq 0 ]; then
  say "ALL 9 CASES GENERATED AND ASSERTED OK"
else
  say "GENERATION HAD FAILURES"; exit 1
fi
