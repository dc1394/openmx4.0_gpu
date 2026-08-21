#!/bin/bash
# RTX 5080 accuracy campaign generator (rtx5080_procedure.md R4; plan v2.6
# sec. 7.5-1 / 13.2-B / 13.3): displaced Si 216 (atom 1 +0.05 Ang along x,
# FRAC dx = 0.05/16.293 = 0.00306880), the four dense-solver paths, run to
# CONVERGENCE (scf.maxIter 60, criterion 1e-13 -- the fixed-25 truncation is
# a timing device, not appropriate here).
#
#   s5a_<sys>_<cfg><rep>   sys: bcol|bnc|ccol|cnc   cfg: c16|o|g
#
# Decks derive from the corresponding s5p_<sys>216_<cfg>11 production deck
# (so the eigen.lib / gemmul8 lines are already per-cfg correct); job
# scripts are the same run's script with names and timeout substituted.
# Reps mirror the Pegasus thesis set: every sys x cfg once (rep 11), plus
# bnc o12/g12 for the same-backend run-to-run baseline.
#
# Usage from work/:  ./gen_s5a.sh bcol:c16:11 bnc:o:11 bnc:o:12 ...
set -u
cd "$(dirname "$0")" || exit 1

DX=0.00306880   # 0.05 Ang / 16.293 Ang (conv 3x3x3 cell) in FRAC

fail=0; ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

tmo_for() {  # $1=sys $2=cfg  -- convergence runs need ~60/25 of production
  case "$2" in
    c16) case "$1" in bnc) echo 16200;; cnc) echo 9000;; bcol) echo 7200;; ccol) echo 5400;; esac;;
    *)   case "$1" in bnc) echo 9000;;  cnc) echo 6600;; bcol) echo 3600;; ccol) echo 3600;; esac;;
  esac
}

for pt in "$@"; do
  IFS=: read -r sys cfg rep <<<"$pt"
  case "$sys" in bcol|bnc|ccol|cnc) : ;; *) say "FAIL bad sys $sys"; fail=1; continue;; esac
  case "$cfg" in c16|o|g) : ;; *) say "FAIL bad cfg $cfg"; fail=1; continue;; esac
  src="s5p_${sys}216_${cfg}11"
  [ -s "$src/$src.dat" ] && [ -s "$src/$src.sh" ] || { say "FAIL missing source case $src"; fail=1; continue; }
  c="s5a_${sys}_${cfg}${rep}"
  ngen=$((ngen+1)); mkdir -p "$c"

  # ---- deck: displace atom 1, converge ---------------------------------
  awk -v name="$c" -v dx="$DX" '
    $1=="System.Name"   { print "System.Name                     " name; next }
    $1=="scf.maxIter"   { print "scf.maxIter                 60          # converge, do not truncate"; next }
    $1=="scf.criterion" { print "scf.criterion             1.0e-13      # convergence run"; next }
    /<Atoms.SpeciesAndCoordinates/ { inblk=1; print; next }
    /Atoms.SpeciesAndCoordinates>/ { inblk=0; print; next }
    inblk && $1=="1" { printf "  %4d   Si  %12.8f %12.8f %12.8f   2.0  2.0\n", 1, $3+dx, $4, $5; next }
    { print }
  ' "$src/$src.dat" > "$c/$c.dat"

  # ---- job script: same conventions, renamed + retimed ------------------
  tmo=$(tmo_for "$sys" "$cfg")
  sed -e "s/${src}/${c}/g" \
      -e "s/--timeout [0-9]*/--timeout ${tmo}/" \
      -e "s/^# Fixed 25 SCF,/# Convergence run (up to 60 SCF, criterion 1e-13); displaced Si:\n# atom 1 moved +0.05 Ang along x so forces are non-zero (plan 7.5-1)./" \
      "$src/$src.sh" > "$c/$c.sh"
  chmod +x "$c/$c.sh"
  [ -s "$src/mps_node.sh" ] && cp "$src/mps_node.sh" "$c/mps_node.sh" && chmod +x "$c/mps_node.sh"

  # ---- assertions --------------------------------------------------------
  say "== $c (timeout ${tmo}s)"
  need "System.Name" "$c"  "$(awk '$1=="System.Name"{print $2}' "$c/$c.dat")"
  need "maxIter"     "60"  "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
  need "criterion"   "1.0e-13" "$(awk '$1=="scf.criterion"{print $2}' "$c/$c.dat")"
  case "$cfg" in c16) explib=elpa2;; *) explib=gpusolver;; esac
  need "eigen.lib"   "$explib" "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
  case "$cfg" in o) expg8=1;; *) expg8=0;; esac
  need "gemmul8 line" "$expg8" "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")"
  # displaced x = 0.04166667 + DX
  need "atom1 x"     "0.04473547" "$(sed -n '/<Atoms.SpeciesAndCoordinates/,/Atoms.SpeciesAndCoordinates>/p' "$c/$c.dat" | awk '$1=="1"{print $3}')"
  need "atom2 x"     "0.20833333" "$(sed -n '/<Atoms.SpeciesAndCoordinates/,/Atoms.SpeciesAndCoordinates>/p' "$c/$c.dat" | awk '$1=="2"{print $3}')"
  need "CASE var"    "CASE=${c}" "$(grep -m1 '^CASE=' "$c/$c.sh")"
  need "no stale name" "0" "$(grep -c "$src" "$c/$c.sh")"
  bash -n "$c/$c.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
done

if [ $fail -eq 0 ]; then say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"; else say "GENERATION HAD FAILURES"; exit 1; fi
