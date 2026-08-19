#!/bin/bash
# Si diamond CLUSTER-solver size preflight (research plan v2.6, step 10 of
# sec. 18): the same ladder, gates and case kinds as the band search
# (gen_siband_search.sh), but scf.EigenvalueSolver cluster, Gamma-only.
#
#   sic_<mode><atoms>_<kind>   kind: p  = 3-SCF cuBLAS probe (gemmul8 off)
#                                    p8 = 3-SCF gemmul8-on probe
#                                    o  = 25-SCF confirmation, gemmul8 off
#                                    g  = 25-SCF confirmation, gemmul8 on
#
# Decks are derived from the band-search decks sib_<mode><atoms>_p (same
# frozen geometry, same PAO/settings) with three edits: System.Name,
# scf.EigenvalueSolver Band -> cluster, scf.Kgrid 2 2 2 -> 1 1 1.
# Job scripts from the proven dia64dc_1n_g1 GPU+MPS template, 48 ranks.
#
# Engagement evidence differs from band: the cluster solver prints only
# NEGATIVE banners (<Cluster_DFT_*> fallback/disabled lines) -- feasible
# runs must show ZERO of those, plus the <Set_Hamiltonian> GPU device line
# and 100% .smi utilisation.
#
# Usage from work/:  ./gen_sicluster_search.sh col:216:p nc:300:p ...
set -u
cd "$(dirname "$0")" || exit 1

TPL=dia64dc_1n_g1
[ -s "$TPL/$TPL.sh" ] || { echo "FATAL: template $TPL missing"; exit 1; }

fail=0; ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

elap_for() {  # $1=mode $2=kind  (cluster does 1 dense solve per SCF iter,
  case "$1:$2" in # band did 8; band probes took 40-95 s, so 40 min is ample)
    *:p|*:p8)   echo 00:40:00;;
    col:o|col:g) echo 00:40:00;;
    nc:o|nc:g)  echo 01:00:00;;
  esac
}
secs() { IFS=: read -r h m s <<<"$1"; echo $((10#$h*3600 + 10#$m*60 + 10#$s)); }

header() {  # $1=case $2=mode $3=atoms $4=kind $5=scfmax
  local dimn=$((13 * $3)) what
  case "$4" in
    p)  what="gemmul8 OFF -> cuBLAS FP64 (3-SCF GO/NO-GO probe)";;
    p8) what="gemmul8 on (3-SCF GO/NO-GO probe)";;
    o)  what="gemmul8 OFF -> cuBLAS FP64 (25-SCF confirmation)";;
    g)  what="gemmul8 on (25-SCF confirmation)";;
  esac
  cat <<EOF
# Si diamond CLUSTER-solver size preflight (plan v2.6 step 10): $1.
# Same ladder/gates as the band search: 1 node x 48 ranks x 1 H100, MPS on;
# feasibility = exit 0, no OOM, node peak RSS <= ~102 GiB, and ZERO
# <Cluster_DFT_*> fallback/disabled banners (the cluster GPU path prints
# only negative banners; positive evidence is the <Set_Hamiltonian> GPU
# line plus 100% device utilisation in the .smi).
# System: $3 Si atoms, Si7.0-s2p2d1: col dim n=$dimn / NC dim 2n=$((2*dimn)).
# scf.EigenvalueSolver cluster, Gamma-only (Kgrid 1 1 1), GGA-PBE, 200 Ry,
# scf.maxIter $5, MD.Type Opt x1.  Config: $what.
# Binary: openmx cd5f0d5 + GEMMul8 v3.2.0 (md5 962f8d2519c2e6aa5a6295513f76fee9).
# Single sample: GO/NO-GO only.  A crash or OOM here is a RESULT.
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
  IFS=: read -r mode atoms kind <<<"$pt"
  src="sib_${mode}${atoms}_p"
  [ -s "$src/$src.dat" ] || { say "FAIL missing band deck $src (generate it first)"; fail=1; continue; }
  case "$kind" in p|o) g8=off;; p8|g) g8=on;; *) say "FAIL bad kind $kind"; fail=1; continue;; esac
  case "$kind" in p|p8) scfmax=3;; o|g) scfmax=25;; esac
  c="sic_${mode}${atoms}_${kind}"
  el=$(elap_for "$mode" "$kind"); tmo=$(( $(secs "$el") - 360 ))
  jn="cl_${mode:0:1}${atoms}${kind}"
  ngen=$((ngen+1)); mkdir -p "$c"

  sed -e "s/^System.Name  *${src}\$/System.Name                     ${c}/" \
      -e "s/^\(scf.EigenvalueSolver  *\)Band .*\$/\1cluster     # DC|GDC|Cluster|Band/" \
      -e "s/^\(scf.Kgrid  *\)2 2 2.*\$/\11 1 1       # Gamma-only for the cluster solver/" \
      -e "s/^\(scf.maxIter  *\)3 .*\$/\1${scfmax}          # fixed-iteration timing/" \
      "$src/$src.dat" > "$c/$c.dat"
  if [ "$g8" = on ]; then sed -i '/^scf.gemmul8.enable/d' "$c/$c.dat"; fi

  sed -e "s/${TPL}/${c}/g" \
      -e "s/^#PBS -N .*\$/#PBS -N ${jn}/" \
      -e "s/^#PBS -l elapstim_req=.*\$/#PBS -l elapstim_req=${el}/" \
      -e "s/--timeout 1800/--timeout ${tmo}/" \
      -e "s/MPS on, gemmul8 on/MPS on, gemmul8 ${g8}/" \
      -e "s/of 50 max/of ${scfmax} max/" \
      -e "s/^echo \"=== DC-LNO GPU dispatch lines ===\"\$/echo \"=== Cluster GPU fallback (expect none) + Set_Hamiltonian GPU lines ===\"/" \
      -e "s|^grep -E \"<DC-LNO>\" .*\$|grep -E \"<Cluster_DFT_(Col\|NonCol)>\|<Set_Hamiltonian> GPU device\" \"\$STDOUT\" 2>/dev/null \| head -8|" \
      "${TPL}/${TPL}.sh" > "$c/$c.sh"
  rewrite_header "$c/$c.sh" "$c" "$mode" "$atoms" "$kind" "$scfmax"
  cp "${TPL}/mps_node.sh" "$c/mps_node.sh"

  say "== $c ($el)"
  need "System.Name"  "$c"        "$(awk '$1=="System.Name"{print $2}' "$c/$c.dat")"
  need "solver"       "cluster"   "$(awk '$1=="scf.EigenvalueSolver"{print $2}' "$c/$c.dat")"
  need "Kgrid"        "1 1 1"     "$(awk '$1=="scf.Kgrid"{print $2, $3, $4}' "$c/$c.dat")"
  need "spin"         "$([ "$mode" = nc ] && echo NC || echo off)" "$(awk '$1=="scf.SpinPolarization"{print $2}' "$c/$c.dat")"
  need "eigen.lib"    "gpusolver" "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
  need "scf.maxIter"  "$scfmax"   "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
  need "gemmul8 line" "$([ "$g8" = off ] && echo 1 || echo 0)" "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")"
  need "atoms"        "$atoms"    "$(awk '$1=="Atoms.Number"{print $2}' "$c/$c.dat")"
  need "coord lines"  "$atoms"    "$(awk '/<Atoms.SpeciesAndCoordinates/{f=1;next} /Atoms.SpeciesAndCoordinates>/{f=0} f{n++} END{print n+0}' "$c/$c.dat")"
  need "CASE var"     "CASE=${c}" "$(grep -m1 '^CASE=' "$c/$c.sh")"
  need "PPN"          "PPN=48"    "$(grep -m1 '^PPN=' "$c/$c.sh")"
  need "gpunum"       "1"         "$(grep -c '^#PBS --gpunum-lhost=1' "$c/$c.sh")"
  need "elapstim"     "#PBS -l elapstim_req=${el}" "$(grep -m1 '^#PBS -l elapstim' "$c/$c.sh")"
  need "cluster report" "1"       "$(grep -c 'Cluster_DFT_(Col|NonCol)' "$c/$c.sh")"
  need "no stale refs" "0"        "$(grep -c "dia64dc\|DC-LNO\|${src}/" "$c/$c.sh")"
  need "mps_node.sh"  "yes"       "$([ -s "$c/mps_node.sh" ] && echo yes || echo no)"
  bash -n "$c/$c.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
done

if [ $fail -eq 0 ]; then say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"; else say "GENERATION HAD FAILURES"; exit 1; fi
