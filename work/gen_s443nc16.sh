#!/bin/bash
# Generate the -npernode 16 re-measurement of sidia443_nc_cluster:
#   s443nc_{1,2,3,4}n16_{g,c,d}{1,2,3}
#   g = gpusolver + MPS, gemmul8 on (default)
#   c = elpa2, CPU only, no GPU reservation
#   d = gpusolver2 (distributed ELPA 2026.02 + COSMA), MPS
# 33 cases, not 36: the 1-node d point already exists as s443nc_1n16_d2..4
# (measured 2026-08-15 at exactly this rank count on the same binary, on
# slow-node-free hosts), so it is reused rather than re-run.
#
# Every case is derived from the corresponding 36-rank case by sed and then
# asserted field by field; the header comment block is rewritten wholesale so
# no stale "36 ranks" prose survives into the new scripts.
#
# Walltimes are raised over the 36-rank templates because 16 ranks/node is
# the slower configuration: g 00:40->00:50, c 01:30->02:00, d 01:00->01:15,
# with the mpirun --timeout guards moved out proportionally.
#
# Run from work/:  ./gen_s443nc16.sh
set -u
cd "$(dirname "$0")" || exit 1

fail=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

# 36-rank means (Total/Diagonalization, s) quoted in each new script's header
# so the run has its own reference point without opening the result file.
ref() {
  case "$2$1" in
    g1) echo "183.79 s total / 93.63 diag";;  g2) echo "154.88 s total / 93.75 diag";;
    g3) echo "139.53 s total / 92.44 diag";;  g4) echo "134.30 s total / 91.21 diag";;
    c1) echo "1060.59 s total / 842.71 diag";; c2) echo "671.79 s total / 537.88 diag";;
    c3) echo "501.38 s total / 407.31 diag";; c4) echo "425.17 s total / 349.90 diag";;
    d2) echo "381.56 s total / 325.11 diag";; d3) echo "321.26 s total / 274.15 diag";;
    d4) echo "281.34 s total / 239.66 diag";; *) echo "(none)";;
  esac
}

header() {  # $1 = node count, $2 = cfg -> the replacement comment block
  local n=$1 cfg=$2 what
  case "$cfg" in
    g) what="GPU path (gpusolver, gemmul8 on)";;
    c) what="CPU path (elpa2, no GPU reserved)";;
    d) what="GPU path (gpusolver2 = distributed ELPA 2026.02 + COSMA)";;
  esac
  cat <<EOF
# sidia443_nc_cluster re-measurement at -npernode 16, ${what}.
# One of 33 requests: {1,2,3,4} nodes x {gpusolver+gemmul8-on, gpusolver2,
# elpa2} x 3 reps, all 16 ranks/node on whole reserved nodes (48 cores).
# The 1-node gpusolver2 point is not in this set -- s443nc_1n16_d2..4 already
# measured it at this rank count on this binary and is reused.
# All three paths run on the SAME binary here, the gpusolver2 merge
# (commit f5fb5b3); in the 36-rank table g/o/c came from e227507 instead.
# 384 Si, s2p2d1, non-collinear -> matrix dimension 9984, Cluster solver,
# 25 SCF iterations to 1e-13.
# 36-rank mean for this same point: $(ref "$n" "$cfg").
# A crash here is a result, not a failure of the harness.
EOF
}

# Replace everything between the "#PBS -o" line and "set -u" with a fresh
# header; sed'ing the old prose line by line would leave "36" behind wherever
# the wording differs between the g, c and d templates.
rewrite_header() {  # $1 = file, $2 = node count, $3 = cfg
  local f=$1 hdr
  hdr=$(header "$2" "$3")
  awk -v hdr="$hdr" '
    /^#PBS -o /      { print; print ""; print hdr; skip=1; next }
    skip && /^set -u$/ { skip=0; print "" }
    skip            { next }
    { print }
  ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}

for n in 1 2 3 4; do
  for cfg in g c d; do
    [ "$n" = 1 ] && [ "$cfg" = d ] && continue   # reusing s443nc_1n16_d2..4
    tpl=s443nc_${n}n_${cfg}1
    case "$cfg" in
      g) lib=gpusolver;  old_el=00:40:00; new_el=00:50:00; old_to=1800; new_to=2400;;
      c) lib=elpa2;      old_el=01:30:00; new_el=02:00:00; old_to=4800; new_to=6600;;
      d) lib=gpusolver2; old_el=01:00:00; new_el=01:15:00; old_to=1800; new_to=3600;;
    esac

    for r in 1 2 3; do
      c=s443nc_${n}n16_${cfg}${r}
      mkdir -p "$c"

      sed -e "s/${tpl}/${c}/g" \
          -e "s/^PPN=36\$/PPN=16/" \
          -e "s/^#PBS -N .*\$/#PBS -N s16_${n}${cfg}${r}/" \
          -e "s/elapstim_req=${old_el}/elapstim_req=${new_el}/" \
          -e "s/--timeout ${old_to}/--timeout ${new_to}/" \
          "${tpl}/${tpl}.sh" > "${c}/${c}.sh"
      rewrite_header "${c}/${c}.sh" "$n" "$cfg"

      sed -e "s/${tpl}/${c}/g" "${tpl}/${tpl}.dat" > "${c}/${c}.dat"

      if [ "$cfg" = c ]; then
        rm -f "${c}/mps_node.sh"
      else
        cp "${tpl}/mps_node.sh" "${c}/mps_node.sh"
      fi

      say "== ${c}"
      need "System.Name"   "$c"    "$(awk '$1=="System.Name"{print $2}' "${c}/${c}.dat")"
      need "scf.eigen.lib" "$lib"  "$(awk '$1=="scf.eigen.lib"{print $2}' "${c}/${c}.dat")"
      need "no gemmul8 line" "0"   "$(grep -c 'scf.gemmul8' "${c}/${c}.dat")"
      need "CASE var"  "CASE=${c}" "$(grep -m1 '^CASE=' "${c}/${c}.sh")"
      need "NNODE"     "NNODE=${n}" "$(grep -m1 '^NNODE=' "${c}/${c}.sh")"
      need "PPN"       "PPN=16"    "$(grep -m1 '^PPN=' "${c}/${c}.sh")"
      need "-b nodes"  "#PBS -b ${n}" "$(grep -m1 '^#PBS -b ' "${c}/${c}.sh")"
      need "job name"  "#PBS -N s16_${n}${cfg}${r}" "$(grep -m1 '^#PBS -N ' "${c}/${c}.sh")"
      need "joblog"    "#PBS -o ${c}.joblog" "$(grep -m1 '^#PBS -o ' "${c}/${c}.sh")"
      need "elapstim"  "#PBS -l elapstim_req=${new_el}" "$(grep -m1 '^#PBS -l elapstim' "${c}/${c}.sh")"
      need "timeout"   "1" "$(grep -c -- "--timeout ${new_to}" "${c}/${c}.sh")"
      need "gpunum-lhost" "$([ "$cfg" = c ] && echo 0 || echo 1)" \
                          "$(grep -c '^#PBS --gpunum-lhost=1' "${c}/${c}.sh")"
      need "cpunum-lhost" "#PBS --cpunum-lhost=48" "$(grep -m1 '^#PBS --cpunum' "${c}/${c}.sh")"
      need "no stale refs" "0" "$(grep -c "${tpl}" "${c}/${c}.sh" "${c}/${c}.dat" | awk -F: '{s+=$2} END{print s}')"
      need "no '36' left"  "0" "$(grep -cE 'PPN=36|npernode 36|36 rank' "${c}/${c}.sh")"
      need "mps_node.sh"   "$([ "$cfg" = c ] && echo no || echo yes)" \
                           "$([ -s "${c}/mps_node.sh" ] && echo yes || echo no)"
      bash -n "${c}/${c}.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
    done
  done
done

if [ $fail -eq 0 ]; then
  say "ALL 33 CASES GENERATED AND ASSERTED OK"
else
  say "GENERATION HAD FAILURES"; exit 1
fi
