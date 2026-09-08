#!/bin/bash
# H100 practical-convergence rerun generator -- the Pegasus twin of
# gen_s5a8.sh (thesis tab:practical-convergence verification): same
# displaced Si 216 decks as the siacc accuracy campaign (atom 1
# +0.05 Ang along x), but run with the PRACTICAL criterion under test
# -- scf.criterion 1.0e-8, scf.maxIter 200 -- so OpenMX's own
# convergence logic decides the stop.  DFT.c:1330 tests dUele<criterion
# AND NormRD[0]<10*criterion (SCF_iter>=2), with NormRD[0] the SQUARED
# density residual (the printed <DFT> NormRD is its sqrt), so the second
# condition is a loose artifact guard and the run stops exactly at the
# first arrival of dUele<1e-8 = i_conv^(8).
#
#   siacc8_<sys>_<cfg><rep>   sys: bcol|bnc|ccol|cnc   cfg: c|o|g
#
# Decks derive from the corresponding siacc_<sys>_<cfg>11 convergence
# deck (already displaced; eigen.lib / gemmul8 lines already per-cfg
# correct); job scripts are the same run's script with names substituted
# (timeouts kept -- a 1e-8 stop is strictly earlier than the 1e-13/60-SCF
# parent).  #PBS -N: parent 'a<sys><cfg>11' -> '8<sys><cfg>11'.
#
# Usage from work/:  ./gen_siacc8.sh bcol:c:11 bnc:o:11 ...
set -u
cd "$(dirname "$0")" || exit 1

fail=0; ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

for pt in "$@"; do
  IFS=: read -r sys cfg rep <<<"$pt"
  case "$sys" in bcol|bnc|ccol|cnc) : ;; *) say "FAIL bad sys $sys"; fail=1; continue;; esac
  case "$cfg" in c|o|g) : ;; *) say "FAIL bad cfg $cfg"; fail=1; continue;; esac
  src="siacc_${sys}_${cfg}11"
  [ -s "$src/$src.dat" ] && [ -s "$src/$src.sh" ] || { say "FAIL missing source case $src"; fail=1; continue; }
  c="siacc8_${sys}_${cfg}${rep}"
  ngen=$((ngen+1)); mkdir -p "$c"

  # ---- deck: same displaced geometry, practical criterion ----------------
  awk -v name="$c" '
    $1=="System.Name"   { print "System.Name                     " name; next }
    $1=="scf.maxIter"   { print "scf.maxIter                 200         # practical-convergence rerun"; next }
    $1=="scf.criterion" { print "scf.criterion             1.0e-8       # criterion under test"; next }
    { print }
  ' "$src/$src.dat" > "$c/$c.dat"

  # ---- job script: same conventions, renamed -----------------------------
  sed -e "s/${src}/${c}/g" \
      -e "s/^#PBS -N a${sys}${cfg}11\$/#PBS -N 8${sys}${cfg}11/" \
      -e "s/Runs to$/Practical-convergence rerun:/" \
      -e "s/^# convergence (criterion 1e-13, up to 60 SCF)\./# criterion 1.0e-8, up to 200 SCF; expected stop = i_conv^(8) of the parent./" \
      "$src/$src.sh" > "$c/$c.sh"
  chmod +x "$c/$c.sh"
  [ -s "$src/mps_node.sh" ] && cp "$src/mps_node.sh" "$c/mps_node.sh" && chmod +x "$c/mps_node.sh"

  # ---- assertions --------------------------------------------------------
  say "== $c"
  need "System.Name" "$c"  "$(awk '$1=="System.Name"{print $2}' "$c/$c.dat")"
  need "maxIter"     "200" "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
  need "criterion"   "1.0e-8" "$(awk '$1=="scf.criterion"{print $2}' "$c/$c.dat")"
  case "$cfg" in c) explib=elpa2;; *) explib=gpusolver;; esac
  need "eigen.lib"   "$explib" "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
  case "$cfg" in o) expg8=1;; *) expg8=0;; esac
  need "gemmul8 line" "$expg8" "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")"
  # displaced x carried over from the siacc parent deck
  need "atom1 x"     "0.04473547" "$(sed -n '/<Atoms.SpeciesAndCoordinates/,/Atoms.SpeciesAndCoordinates>/p' "$c/$c.dat" | awk '$1=="1"{print $3}')"
  need "atom2 x"     "0.20833333" "$(sed -n '/<Atoms.SpeciesAndCoordinates/,/Atoms.SpeciesAndCoordinates>/p' "$c/$c.dat" | awk '$1=="2"{print $3}')"
  need "PBS -N"      "8${sys}${cfg}11" "$(awk '/^#PBS -N/{print $3}' "$c/$c.sh")"
  case "$cfg" in c) expgpu=0;; *) expgpu=1;; esac
  need "gpunum line" "$expgpu" "$(grep -c '^#PBS --gpunum-lhost=1' "$c/$c.sh")"
  need "CASE var"    "CASE=${c}" "$(grep -m1 '^CASE=' "$c/$c.sh")"
  need "no stale name" "0" "$(grep -c "$src" "$c/$c.sh")"
  bash -n "$c/$c.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
done

if [ $fail -eq 0 ]; then say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"; else say "GENERATION HAD FAILURES"; exit 1; fi
