#!/bin/bash
# Generate the dia64_dc-lno node-scaling benchmark at -npernode 48:
#   dia64dc_{1,2,3,4}n_{g,o,c}{1,2,3}   (36 cases)
#   g = gpusolver + MPS, scf.gemmul8.enable on (default)
#   o = gpusolver + MPS, scf.gemmul8.enable off (plain cuBLAS FP64)
#   c = elpa2, CPU only, no GPU reserved
#
# The .dat is work/dia64_dc-lno/dia64_dc-lno.dat as edited 2026-08-16 10:51:
# MD.Type Opt with MD.maxIter 1, i.e. one geometry step, so the force
# calculation actually runs.  The earlier dia64dc_1n48_* cases used the same
# deck with MD.Type nomd and are superseded by this campaign; they are left on
# disk but must not be mixed into these means.
#
# Job scripts are derived from dia64dc_1n48_{g,o,c}1, which already carry
# PPN=48, the "of 50 max" SCF count and the <DC-LNO> dispatch report.
#
# Rank-count caveat, baked into each script's header so it travels with the
# run: this system has 64 atoms, and DC-LNO parallelises the dense per-cluster
# eigenproblems over atoms.  Set_Allocate_Atom2CPU.c clamps the atom-owning
# rank count to min(numprocs, atomnum), so at -npernode 48 only the 1-node run
# (48 ranks) has every rank owning a cluster.  From 2 nodes on, 64 ranks own
# one cluster each and the rest own none -- 96, 144 and 192 ranks all do the
# same 64 cluster solves.  Solver==11 does get the ID2ID spreading remap
# (line 333) so those 64 owners are distributed across the nodes rather than
# packed onto the first ones.
#
# A capped owner count does NOT imply a flat curve past 2 nodes, and the
# rep-1 measurements say it is not flat: CPU Diagonalization came out at
# 571.6 / 274.7 / 211.7 / 182.8 s for 1/2/3/4 nodes.  The owners are capped,
# but spreading them thins each node out -- 48 owners on one node at 1 node,
# 16 per node at 4 -- so every owner gets more memory bandwidth and cache, and
# on the GPU path more devices as well.  That is where the gain past 2 nodes
# comes from; it is not additional owners.
#
# Run from work/:  ./gen_dia64dc_nodes.sh                 (all 36)
#                  ./gen_dia64dc_nodes.sh 1:o:4 4:g:5     (only those points)
# The second form exists for replacement reps: when a run draws a slow node it
# is discarded and re-measured under a new rep number, and regenerating all 36
# just to add one case would rewrite 35 scripts that already have joblogs.
set -u
cd "$(dirname "$0")" || exit 1

# selection filter: empty = everything, otherwise a set of "<n>:<cfg>:<rep>"
WANT=" $* "
wanted() {  # $1=n $2=cfg $3=rep
  [ "$WANT" = "  " ] && return 0
  case "$WANT" in *" $1:$2:$3 "*) return 0;; esac
  return 1
}

SRC=dia64_dc-lno/dia64_dc-lno.dat
[ -s "$SRC" ] || { echo "FATAL: $SRC missing"; exit 1; }

# The whole point of this re-run: refuse to generate anything if the deck is
# not doing the single optimisation step that produces forces.
mdtype=$(awk '$1=="MD.Type"{print $2}' "$SRC")
mditer=$(awk '$1=="MD.maxIter"{print $2}' "$SRC")
if [ "$mdtype" != "Opt" ] || [ "$mditer" != "1" ]; then
  echo "FATAL: expected MD.Type=Opt and MD.maxIter=1 in $SRC, got [$mdtype] [$mditer]"
  exit 1
fi

fail=0
ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

header() {  # $1 = node count, $2 = cfg
  local n=$1 what owners
  case "$2" in
    g) what="GPU path (gpusolver + MPS, gemmul8 on)";;
    o) what="GPU path (gpusolver + MPS, gemmul8 OFF -> plain cuBLAS FP64)";;
    c) what="CPU path (elpa2, no GPU reserved)";;
  esac
  owners=$(( n * 48 )); [ "$owners" -gt 64 ] && owners=64
  cat <<EOF
# dia64_dc-lno node scaling, ${n} node(s) x 48 ranks, ${what}.
# One of 36 requests: {1,2,3,4} nodes x {gemmul8-on, gemmul8-off, elpa2} x 3
# reps at -npernode 48, so Total and Diagonalization come with a mean.
# Binary: gpusolver2 merge (commit f5fb5b3).  64 C atoms, C6.0-s2p2d1 -> 832
# basis functions, scf.EigenvalueSolver dc-lno, Gamma-only, 50 SCF max to
# 1e-10, orderN.HoppingRanges 6.0 Ang, MD.Type Opt with MD.maxIter 1 so the
# force calculation runs.
# DC-LNO parallelises over atoms and Set_Allocate_Atom2CPU.c clamps the
# atom-owning ranks to min(numprocs, atomnum): of the $(( n * 48 )) ranks here,
# ${owners} own a cluster.  Extra ranks still carry grid work, and Solver==11
# spreads the owners across nodes, but the per-cluster eigen work does not
# subdivide further.
# DC-LNO sends a cluster to the GPU only when its local matrix dimension is
# >= 800 (Divide_Conquer_LNO.c, OPENMX_DCLNO_GPU_THRESHOLD); the .smi sampler
# is the evidence for whether that happened.
# A crash here is a result, not a failure of the harness.
EOF
}

rewrite_header() {  # $1 = file, $2 = node count, $3 = cfg
  local f=$1 hdr
  hdr=$(header "$2" "$3")
  awk -v hdr="$hdr" '
    /^#PBS -o /        { print; print ""; print hdr; skip=1; next }
    skip && /^set -u$/ { skip=0; print "" }
    skip               { next }
    { print }
  ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}

for n in 1 2 3 4; do
  for cfg in g o c; do
    tpl=dia64dc_1n48_${cfg}1
    case "$cfg" in
      g|o) lib=gpusolver; gpu=1; el=00:40:00;;
      c)   lib=elpa2;     gpu=0; el=01:30:00;;
    esac

    for r in 1 2 3 4 5 6; do
      wanted "$n" "$cfg" "$r" || continue
      [ -z "$*" ] && [ "$r" -gt 3 ] && continue    # default campaign is 3 reps
      c=dia64dc_${n}n_${cfg}${r}
      ngen=$((ngen+1))
      mkdir -p "$c"

      sed -e "s/${tpl}/${c}/g" \
          -e "s/^NNODE=1\$/NNODE=${n}/" \
          -e "s/^#PBS -b 1\$/#PBS -b ${n}/" \
          -e "s/^#PBS -N .*\$/#PBS -N d64_${n}${cfg}${r}/" \
          "${tpl}/${tpl}.sh" > "${c}/${c}.sh"
      rewrite_header "${c}/${c}.sh" "$n" "$cfg"

      sed -e "s/^\(System.Name  *\)dia64_dc-lno\$/\1${c}/" "$SRC" > "${c}/${c}.dat"
      if [ "$cfg" = c ]; then
        sed -i -e "s/^\(scf.eigen.lib  *\)gpusolver\$/\1elpa2/" "${c}/${c}.dat"
      elif [ "$cfg" = o ]; then
        sed -i -e "s/^\(scf.eigen.lib  *gpusolver\)\$/\1\nscf.gemmul8.enable        off       # A\/B: plain cuBLAS FP64/" "${c}/${c}.dat"
      fi

      if [ "$cfg" = c ]; then rm -f "${c}/mps_node.sh"; else cp "${tpl}/mps_node.sh" "${c}/mps_node.sh"; fi

      say "== ${c}"
      need "System.Name"    "$c"      "$(awk '$1=="System.Name"{print $2}' "${c}/${c}.dat")"
      need "solver"     "dc-lno"      "$(awk '$1=="scf.EigenvalueSolver"{print $2}' "${c}/${c}.dat")"
      need "scf.eigen.lib"  "$lib"    "$(awk '$1=="scf.eigen.lib"{print $2}' "${c}/${c}.dat")"
      need "MD.Type"        "Opt"     "$(awk '$1=="MD.Type"{print $2}' "${c}/${c}.dat")"
      need "MD.maxIter"     "1"       "$(awk '$1=="MD.maxIter"{print $2}' "${c}/${c}.dat")"
      need "gemmul8 line"   "$([ "$cfg" = o ] && echo 1 || echo 0)" \
                            "$(grep -c '^scf.gemmul8.enable' "${c}/${c}.dat")"
      need "gemmul8 value"  "$([ "$cfg" = o ] && echo off || echo -)" \
                            "$(awk '$1=="scf.gemmul8.enable"{print $2}' "${c}/${c}.dat" | grep . || echo -)"
      need "atoms"          "64"      "$(awk '$1=="Atoms.Number"{print $2}' "${c}/${c}.dat")"
      need "DATA.PATH" "../../DFT_DATA19" "$(awk '$1=="DATA.PATH"{print $2}' "${c}/${c}.dat")"
      need "CASE var"   "CASE=${c}"   "$(grep -m1 '^CASE=' "${c}/${c}.sh")"
      need "NNODE"      "NNODE=${n}"  "$(grep -m1 '^NNODE=' "${c}/${c}.sh")"
      need "PPN"        "PPN=48"      "$(grep -m1 '^PPN=' "${c}/${c}.sh")"
      need "-b nodes"   "#PBS -b ${n}" "$(grep -m1 '^#PBS -b ' "${c}/${c}.sh")"
      need "job name"   "#PBS -N d64_${n}${cfg}${r}" "$(grep -m1 '^#PBS -N ' "${c}/${c}.sh")"
      need "joblog"     "#PBS -o ${c}.joblog" "$(grep -m1 '^#PBS -o ' "${c}/${c}.sh")"
      need "elapstim"   "#PBS -l elapstim_req=${el}" "$(grep -m1 '^#PBS -l elapstim' "${c}/${c}.sh")"
      need "cpunum-lhost" "#PBS --cpunum-lhost=48" "$(grep -m1 '^#PBS --cpunum' "${c}/${c}.sh")"
      need "gpunum-lhost" "$gpu" "$(grep -c '^#PBS --gpunum-lhost=1' "${c}/${c}.sh")"
      need "no stale refs" "0" "$(grep -c "${tpl}\|s443nc\|1n48" "${c}/${c}.sh" "${c}/${c}.dat" | awk -F: '{s+=$2} END{print s}')"
      need "DC-LNO report line" "$([ "$cfg" = c ] && echo 0 || echo 1)" \
                                "$(grep -cF 'grep -E "<DC-LNO>"' "${c}/${c}.sh")"
      need "mps_node.sh" "$([ "$cfg" = c ] && echo no || echo yes)" \
                         "$([ -s "${c}/mps_node.sh" ] && echo yes || echo no)"
      bash -n "${c}/${c}.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
    done
  done
done

if [ $fail -eq 0 ]; then
  say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"
else
  say "GENERATION HAD FAILURES"; exit 1
fi
