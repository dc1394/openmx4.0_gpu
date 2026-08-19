#!/bin/bash
# Accuracy campaign (research plan v2.6 sec. 7.5 / 13.2-B, step 16):
#
#   siacc_<sys>_<cfg><rep>
#     sys: bcol/bnc/ccol/cnc = DISPLACED Si, 216 atoms (atom 1 moved
#            +0.05 Ang along x -> non-zero forces), the four dense-solver
#            paths; decks derive from the sip_ production decks with
#            scf.maxIter 60 and scf.criterion back to 1e-13 (these runs
#            must CONVERGE; fixed-25 truncation is a timing device).
#          mno  = Crys-MnO.dat (collinear AFM, band 5x5x5) -- energy,
#                 force, spin moments across configs.
#          cr2  = Cr2.dat  (NC + SOC on, cluster)  } bundled examples;
#          gaas = GaAs.dat (NC + SOC on, band)     } dense dims are far
#                 below the GPU dispatch thresholds, so these validate
#                 correctness/fallback sanity of the three configs, NOT
#                 GPU numerics (stated in the header of each run).
#     cfg: c = CPU elpa2 flat 48; o = GPU cuBLAS + MPS; g = GPU GEMMul8 + MPS
#
# Deviations from the bundled examples (documented here once):
#   DATA.PATH -> ../../DFT_DATA19; MD.Type -> Opt with MD.maxIter 1 (site
#   convention: the force step must actually run); Crys-MnO scf.maxIter
#   3 -> 100 (the example is a 3-iter smoke test; accuracy needs
#   convergence); explicit scf.eigen.lib/gemmul8 lines appended per cfg.
#
# Acceptance targets (plan 13.3): |dE| <= 1e-10 Ha, |dF|max <= 1e-6
# Ha/Bohr between configs, judged against the run-to-run baseline.
#
# Usage from work/:  ./gen_siacc.sh dsi:bcol:o:1 mno:c:1 cr2:g:1 ...
set -u
cd "$(dirname "$0")" || exit 1

GTPL=dia64dc_1n_g1
CTPL=dia64dc_1n_c1
DX=0.00306880   # 0.05 Ang / 16.293 Ang (conv 3x3x3 cell) in FRAC

fail=0; ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

header() {  # $1=case $2=sys $3=cfg
  local what note=""
  case "$3" in
    c) what="CPU elpa2, flat MPI 48 ranks, no GPU";;
    o) what="GPU gpusolver + MPS, gemmul8 off (cuBLAS FP64)";;
    g) what="GPU gpusolver + MPS, gemmul8 on (INT8 FP64 emulation)";;
  esac
  case "$2" in
    bcol|bnc|ccol|cnc)
      note="Displaced Si: 216 atoms, atom 1 moved +0.05 Ang along x, so
# forces are non-zero and DISCRIMINATE the configs (plan 7.5-1).  Runs to
# convergence (criterion 1e-13, up to 60 SCF).";;
    mno)  note="Crys-MnO bundled example: collinear AFM MnO, band 5x5x5
# (plan 7.5-2: energy, force, spin moment).  Dense dims are tiny; the GPU
# build's dense path may fall back below its dispatch threshold -- this
# checks correctness across configs, not GPU numerics.";;
    cr2)  note="Cr2 bundled example: NON-COLLINEAR + spin-orbit ON, cluster
# (plan 7.5-3).  2 atoms: dense dims are far below GPU thresholds; this
# checks correctness/fallback sanity across configs, not GPU numerics.";;
    gaas) note="GaAs bundled example: NON-COLLINEAR + spin-orbit ON, band
# (plan 7.5-3).  2 atoms: dense dims are far below GPU thresholds; this
# checks correctness/fallback sanity across configs, not GPU numerics.";;
  esac
  cat <<EOF
# Accuracy campaign (plan v2.6 sec. 7.5 / 13.2-B): $1.
# $note
# Config: $what.  48 ranks, -nt 1, whole node.
# Compare across configs: Utot, SCF count, chemical potential, the full
# coordinates.forces block, and spin moments where magnetic.
# Targets (plan 13.3): |dE| <= 1e-10 Ha, |dF|max <= 1e-6 Ha/Bohr.
# Binary: work/openmx as deployed at submission; the authoritative build
# identity (commit/tag/md5) is recorded in <case>.manifest.json.
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

mkdeck_dsi() {  # $1=sys(bcol..) $2=cfg $3=outfile
  local srccfg src
  case "$2" in c) srccfg=c481;; o) srccfg=o1;; g) srccfg=g1;; esac
  src="sip_${1}216_${srccfg}"
  [ -s "$src/$src.dat" ] || { say "FAIL missing $src deck"; fail=1; return 1; }
  awk -v name="$4" -v dx="$DX" '
    $1=="System.Name"   { print "System.Name                     " name; next }
    $1=="scf.maxIter"   { print "scf.maxIter                 60          # converge, do not truncate"; next }
    $1=="scf.criterion" { print "scf.criterion             1.0e-13      # convergence run"; next }
    /<Atoms.SpeciesAndCoordinates/ { inblk=1; print; next }
    /Atoms.SpeciesAndCoordinates>/ { inblk=0; print; next }
    inblk && $1=="1" { printf "  %4d   Si  %12.8f %12.8f %12.8f   2.0  2.0\n", 1, $3+dx, $4, $5; next }
    { print }
  ' "$src/$src.dat" > "$3"
}

mkdeck_example() {  # $1=sys $2=cfg $3=outfile $4=case
  local src
  case "$1" in
    mno)  src=large3_example/Crys-MnO.dat;;
    cr2)  src=input_example/Cr2.dat;;
    gaas) src=input_example/GaAs.dat;;
  esac
  [ -s "$src" ] || { say "FAIL missing $src"; fail=1; return 1; }
  awk -v name="$4" -v sys="$1" -v cfg="$2" '
    $1=="System.CurrrentDirectory" { print; next }
    $1=="System.Name"   { print "System.Name                     " name; next }
    $1=="DATA.PATH"     { print "DATA.PATH                       ../../DFT_DATA19"; seen_dp=1; next }
    $1=="scf.maxIter" && sys=="mno" { print "scf.maxIter                100        # example had 3 (smoke test); accuracy needs convergence"; next }
    $1=="MD.Type"       { print "MD.Type                     Opt        # site convention: run the force step"; next }
    $1=="MD.maxIter"    { print "MD.maxIter                    1"; md=1; next }
    $1=="scf.EigenvalueSolver" {
      print
      if (cfg=="c") print "scf.eigen.lib             elpa2"
      else          print "scf.eigen.lib             gpusolver"
      if (cfg=="o") print "scf.gemmul8.enable        off       # A/B: plain cuBLAS FP64"
      next
    }
    $1=="scf.eigen.lib" || $1=="scf.gemmul8.enable" { next }  # drop any preset
    { print }
    END {
      if (!seen_dp) print "DATA.PATH                       ../../DFT_DATA19"
      if (!md)      print "MD.maxIter                    1"
    }
  ' "$src" > "$3"
}

for pt in "$@"; do
  IFS=: read -r k1 k2 k3 k4 <<<"$pt"
  if [ "$k1" = dsi ]; then
    kind=dsi; sys=$k2; cfg=$k3; rep=$k4
  else
    kind=$k1; sys=$k1; cfg=$k2; rep=$k3
  fi
  case "$cfg" in c|o|g) : ;; *) say "FAIL bad cfg in $pt"; fail=1; continue;; esac
  c="siacc_${sys}_${cfg}${rep}"; jn="a${sys}${cfg}${rep}"
  ngen=$((ngen+1)); mkdir -p "$c"

  if [ "$kind" = dsi ]; then
    mkdeck_dsi "$sys" "$cfg" "$c/$c.dat" "$c" || continue
  else
    mkdeck_example "$sys" "$cfg" "$c/$c.dat" "$c" || continue
  fi

  case "$kind:$cfg" in
    dsi:c) el=01:30:00;; dsi:*) el=01:00:00;;
    *:c)   el=00:40:00;; *)     el=00:40:00;;
  esac
  tmo=$(( $( IFS=:; set -- $el; echo $((10#$1*3600+10#$2*60+10#$3)) ) - 360 ))

  if [ "$cfg" = c ]; then
    sed -e "s/${CTPL}/${c}/g" \
        -e "s/^#PBS -N .*\$/#PBS -N ${jn}/" \
        -e "s/^#PBS -l elapstim_req=.*\$/#PBS -l elapstim_req=${el}/" \
        -e "s/--timeout [0-9]*/--timeout ${tmo}/" \
        -e "s/of 50 max/of 60-100 max/" \
        "${CTPL}/${CTPL}.sh" > "$c/$c.sh"
  else
    sed -e "s/${GTPL}/${c}/g" \
        -e "s/^#PBS -N .*\$/#PBS -N ${jn}/" \
        -e "s/^#PBS -l elapstim_req=.*\$/#PBS -l elapstim_req=${el}/" \
        -e "s/--timeout 1800/--timeout ${tmo}/" \
        -e "s/MPS on, gemmul8 on/MPS on, gemmul8 $([ "$cfg" = g ] && echo on || echo off)/" \
        -e "s/of 50 max/of 60-100 max/" \
        -e "s/^echo \"=== DC-LNO GPU dispatch lines ===\"\$/echo \"=== GPU dispatch \/ fallback lines ===\"/" \
        -e "s|^grep -E \"<DC-LNO>\" .*\$|grep -E \"<Band_DFT_Col>\|<Band_DFT_NonCol>\|<Band>\|<Cluster_DFT_(Col\|NonCol)>\|<Set_Hamiltonian> GPU device\" \"\$STDOUT\" 2>/dev/null \| head -8|" \
        "${GTPL}/${GTPL}.sh" > "$c/$c.sh"
    cp "${GTPL}/mps_node.sh" "$c/mps_node.sh"
  fi
  rewrite_header "$c/$c.sh" "$c" "$sys" "$cfg"

  say "== $c ($el)"
  need "System.Name" "$c" "$(awk '$1=="System.Name"{print $2}' "$c/$c.dat")"
  case "$cfg" in
    c) need "eigen.lib" "elpa2" "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
       need "no gemmul8" "0" "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")";;
    o) need "eigen.lib" "gpusolver" "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
       need "gemmul8 off" "off" "$(awk '$1=="scf.gemmul8.enable"{print $2}' "$c/$c.dat")";;
    g) need "eigen.lib" "gpusolver" "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
       need "no gemmul8 line" "0" "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")";;
  esac
  need "MD.Type Opt" "Opt" "$(awk '$1=="MD.Type"{print $2}' "$c/$c.dat")"
  need "MD.maxIter 1" "1"  "$(awk '$1=="MD.maxIter"{print $2}' "$c/$c.dat")"
  need "DATA.PATH" "../../DFT_DATA19" "$(awk '$1=="DATA.PATH"{print $2}' "$c/$c.dat")"
  if [ "$kind" = dsi ]; then
    need "displaced x" "0.04473547" "$(awk '/<Atoms.SpeciesAndCoordinates/{f=1;next} /Atoms.SpeciesAndCoordinates>/{f=0} f&&$1=="1"{printf "%.8f",$3}' "$c/$c.dat")"
    need "maxIter 60" "60" "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
    need "criterion" "1.0e-13" "$(awk '$1=="scf.criterion"{print $2}' "$c/$c.dat")"
    need "atoms 216" "216" "$(awk '$1=="Atoms.Number"{print $2}' "$c/$c.dat")"
  fi
  need "CASE var" "CASE=${c}" "$(grep -m1 '^CASE=' "$c/$c.sh")"
  need "elapstim" "#PBS -l elapstim_req=${el}" "$(grep -m1 '^#PBS -l elapstim' "$c/$c.sh")"
  need "no stale refs" "0" "$(grep -c "dia64dc\|DC-LNO" "$c/$c.sh")"
  bash -n "$c/$c.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
done

if [ $fail -eq 0 ]; then say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"; else say "GENERATION HAD FAILURES"; exit 1; fi
