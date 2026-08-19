#!/bin/bash
# Si diamond band-solver size search (research plan v2.6, section 7.3):
# generate 1-node x 48-rank GPU probe / confirmation cases on the fixed
# candidate ladder of near-isotropic Si supercells built from ../sidia.dat.
#
#   candidate ladder (atoms : family : multiplicity : col dim n / NC dim 2n)
#     216  conv 3x3x3   2808 /  5616      (cubic,  16.293 Ang)
#     250  prim 5x5x5   3250 /  6500      (fcc rhombohedron, 19.202 Ang, 60 deg)
#     288  conv 4x3x3   3744 /  7488
#     300  prim 6x5x5   3900 /  7800
#     360  prim 6x6x5   4680 /  9360
#     384  conv 4x4x3   4992 /  9984
#     432  prim 6x6x6   5616 / 11232      (fcc rhombohedron, 23.042 Ang)
#     512  conv 4x4x4   6656 / 13312      (cubic,  21.724 Ang)
#
# conv = diagonal multiple of the 8-atom conventional cubic cell (a=5.431 Ang)
#        taken verbatim from ../sidia.dat;
# prim = m1xm2xm3 multiple of the 2-atom fcc primitive cell (a/2 lattice
#        vectors, basis 0 and 1/4,1/4,1/4).  Mixing families is fine for the
#        SEARCH ladder; the scaling series later uses one fixed input, and the
#        plan itself orders a boundary-neighbour re-check because RSS need not
#        be strictly monotone across cell shapes.
#
# Case kinds (research plan 7.3.2):
#   p   = 3-SCF + force probe, gpusolver + MPS, scf.gemmul8.enable off (cuBLAS)
#   p8  = same probe with gemmul8 on (default; no gemmul8 line, step 5)
#   o   = 25-SCF confirmation, gemmul8 off (step 6)
#   g   = 25-SCF confirmation, gemmul8 on  (step 6)
# Probes are single samples: GO/NO-GO on memory and path only, never a timing
# ranking.  Feasibility gate: exit 0, no OOM, no "cannot fit" fallback banner,
# node peak RSS <= 80% of 128 GiB (~102 GiB).
#
# Usage from work/:
#   ./gen_siband_search.sh                          # wave 1: col/nc endpoints
#   ./gen_siband_search.sh col:288:p nc:300:p8 ...  # any specific points
# Case name: sib_<mode><atoms>_<kind>, e.g. sib_col512_p, sib_nc384_c25g is
# spelled sib_nc384_g (kind letter only).
set -u
cd "$(dirname "$0")" || exit 1

SRC=../sidia.dat
TPL=dia64dc_1n_g1          # proven 1-node GPU+MPS job script to derive from
[ -s "$SRC" ] || { echo "FATAL: $SRC missing"; exit 1; }
[ -s "$TPL/$TPL.sh" ] || { echo "FATAL: template $TPL missing"; exit 1; }

# The search runs the plan's fixed-25-SCF deck truncated to 3 SCF for probes;
# the deck must carry the force-producing single Opt step throughout.
mdtype=$(awk '$1=="MD.Type"{print $2}' "$SRC")
mditer=$(awk '$1=="MD.maxIter"{print $2}' "$SRC")
if [ "$mdtype" != "Opt" ] || [ "$mditer" != "1" ]; then
  echo "FATAL: expected MD.Type=Opt and MD.maxIter=1 in $SRC, got [$mdtype] [$mditer]"
  exit 1
fi

A=5.4310000   # conventional lattice constant from sidia.dat (Ang)

# ladder: atoms family m1 m2 m3
LADDER="
216 conv 3 3 3
250 prim 5 5 5
288 conv 4 3 3
300 prim 6 5 5
360 prim 6 6 5
384 conv 4 4 3
432 prim 6 6 6
512 conv 4 4 4
"

# walltimes / mpirun timeouts per (mode,kind); timeout = elapstim - 6 min
elap_for() {  # $1=mode $2=kind
  case "$1:$2" in
    col:p|col:p8) echo 01:00:00;;
    nc:p|nc:p8)   echo 01:40:00;;
    # 25-SCF walltimes tightened after the 3-SCF probes measured: col216
    # <~224 s and nc250 <~450 s estimated for 25 SCF; 40/60 min is still
    # 5-8x headroom.
    col:o|col:g)  echo 00:40:00;;
    nc:o|nc:g)    echo 01:00:00;;
  esac
}
secs() { IFS=: read -r h m s <<<"$1"; echo $((10#$h*3600 + 10#$m*60 + 10#$s)); }

fail=0; ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

lookup() {  # $1=atoms -> "family m1 m2 m3"
  echo "$LADDER" | awk -v a="$1" '$1==a {print $2, $3, $4, $5}'
}

emit_coords() {  # $1=family $2=m1 $3=m2 $4=m3 -> stdout frac coord lines
  if [ "$1" = conv ]; then
    awk -v n1="$2" -v n2="$3" -v n3="$4" '
      /<Atoms.SpeciesAndCoordinates/ {f=1; next}
      /Atoms.SpeciesAndCoordinates>/ {f=0}
      f { m++; x[m]=$3; y[m]=$4; z[m]=$5 }
      END {
        id=0
        for (i=0;i<n1;i++) for (j=0;j<n2;j++) for (k=0;k<n3;k++)
          for (b=1;b<=m;b++)
            printf "  %4d   Si  %12.8f %12.8f %12.8f   2.0  2.0\n",
                   ++id, (x[b]+i)/n1, (y[b]+j)/n2, (z[b]+k)/n3
      }' "$SRC"
  else
    awk -v m1="$2" -v m2="$3" -v m3="$4" 'BEGIN {
      id=0; b[1]=0.0; b[2]=0.25
      for (i=0;i<m1;i++) for (j=0;j<m2;j++) for (k=0;k<m3;k++)
        for (t=1;t<=2;t++)
          printf "  %4d   Si  %12.8f %12.8f %12.8f   2.0  2.0\n",
                 ++id, (i+b[t])/m1, (j+b[t])/m2, (k+b[t])/m3
    }'
  fi
}

emit_cell() {  # $1=family $2=m1 $3=m2 $4=m3
  if [ "$1" = conv ]; then
    awk -v a="$A" -v n1="$2" -v n2="$3" -v n3="$4" 'BEGIN {
      printf "  %13.7f %13.7f %13.7f\n", a*n1, 0, 0
      printf "  %13.7f %13.7f %13.7f\n", 0, a*n2, 0
      printf "  %13.7f %13.7f %13.7f\n", 0, 0, a*n3
    }'
  else
    awk -v a="$A" -v m1="$2" -v m2="$3" -v m3="$4" 'BEGIN {
      h = a/2
      printf "  %13.7f %13.7f %13.7f\n", 0,     m1*h, m1*h
      printf "  %13.7f %13.7f %13.7f\n", m2*h,  0,    m2*h
      printf "  %13.7f %13.7f %13.7f\n", m3*h,  m3*h, 0
    }'
  fi
}

header() {  # $1=case $2=mode $3=atoms $4=kind $5=family $6..8=m1m2m3 $9=scfmax
  local dimn=$((13 * $3)) g8what scfmax=$9
  case "$4" in
    p)  g8what="gemmul8 OFF -> plain cuBLAS FP64 (3-SCF GO/NO-GO probe)";;
    p8) g8what="gemmul8 on, default (3-SCF GO/NO-GO probe, plan step 5)";;
    o)  g8what="gemmul8 OFF -> plain cuBLAS FP64 (25-SCF confirmation, step 6)";;
    g)  g8what="gemmul8 on (25-SCF confirmation, step 6)";;
  esac
  cat <<EOF
# Si diamond BAND-solver size search (research plan v2.6 sec. 7.3): $1.
# Purpose: find the largest candidate that runs SAFELY at 1 node x 48 ranks
# x 1 H100 -- feasibility gate is exit 0, no OOM, no "cannot fit" GPU
# fallback banner, node peak RSS <= ~102 GiB (80% of 128).  Single sample:
# GO/NO-GO only, never a timing ranking.
# System: $3 Si atoms ($5 supercell $6x$7x$8 of sidia.dat, a=5.431 Ang),
# Si7.0-s2p2d1 -> 13 basis/atom: col dim n=$dimn, NC dim 2n=$((2*dimn)).
# scf.EigenvalueSolver Band, scf.Kgrid 2x2x2 (8 computed k points), GGA-PBE,
# 200 Ry, scf.maxIter $scfmax, criterion 1e-13 (fixed-iteration timing deck),
# MD.Type Opt with MD.maxIter 1 so the force step runs.  Mode: $2.
# Config: gpusolver + MPS, $g8what.
# Binary: gpusolver2 merge f5fb5b3 -- preflight only; paper production runs
# will be re-measured from the v2.0_thesis tag once instrumentation lands.
# Engagement evidence: "<Band_DFT_Col>/<Band_DFT_NonCol> GPU device" banner
# plus the .smi sampler; a "cannot fit" line means the dense GPU path fell
# back and the probe does NOT count as feasible.
# A crash or OOM here is a RESULT (an infeasible size), not a harness bug.
EOF
}

rewrite_header() {  # $1=file $2..=header args
  local f=$1; shift
  local hdr; hdr=$(header "$@")
  awk -v hdr="$hdr" '
    /^#PBS -o /        { print; print ""; print hdr; skip=1; next }
    skip && /^set -u$/ { skip=0; print "" }
    skip               { next }
    { print }
  ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}

# fixed candidate table, written once so the set is frozen before measurement
if [ ! -s siband_candidates.txt ]; then
  {
    echo "# Si diamond band-search candidate ladder, frozen $(date '+%Y-%m-%d %H:%M')"
    echo "# from sidia.dat (8-atom conventional cell, a=5.431 Ang, Si7.0-s2p2d1)"
    echo "# atoms family m1 m2 m3   n=13N   2n=26N"
    echo "$LADDER" | awk 'NF {printf "  %4d  %s  %d %d %d   %6d  %6d\n", $1,$2,$3,$4,$5, 13*$1, 26*$1}'
  } > siband_candidates.txt
  say "wrote siband_candidates.txt"
fi

[ $# -gt 0 ] && POINTS="$*" || POINTS="col:216:p col:512:p nc:216:p nc:384:p"

for pt in $POINTS; do
  IFS=: read -r mode atoms kind <<<"$pt"
  geom=$(lookup "$atoms")
  [ -n "$geom" ] || { say "FAIL unknown ladder atom count: $atoms"; fail=1; continue; }
  read -r family m1 m2 m3 <<<"$geom"
  case "$mode" in col) spin=off;; nc) spin=NC;; *) say "FAIL bad mode $mode"; fail=1; continue;; esac
  case "$kind" in p|o) g8=off;; p8|g) g8=on;; *) say "FAIL bad kind $kind"; fail=1; continue;; esac
  case "$kind" in p|p8) scfmax=3;; o|g) scfmax=25;; esac

  c="sib_${mode}${atoms}_${kind}"
  el=$(elap_for "$mode" "$kind"); tmo=$(( $(secs "$el") - 360 ))
  jn="sb_${mode:0:1}${atoms}${kind}"
  ngen=$((ngen+1))
  mkdir -p "$c"

  emit_coords "$family" "$m1" "$m2" "$m3" > "$c/.coords.tmp"
  emit_cell   "$family" "$m1" "$m2" "$m3" > "$c/.cell.tmp"

  awk -v name="$c" -v natom="$atoms" -v spin="$spin" -v scfmax="$scfmax" -v g8="$g8" \
      -v coordf="$c/.coords.tmp" -v cellf="$c/.cell.tmp" '
    $1=="System.Name"          { print "System.Name                     " name; next }
    $1=="Atoms.Number"         { print "Atoms.Number                     " natom; next }
    $1=="scf.SpinPolarization" { print "scf.SpinPolarization        " spin "        # On|Off|NC"; next }
    $1=="scf.maxIter"          { print "scf.maxIter                 " scfmax "          # fixed-iteration timing"; next }
    $1=="scf.eigen.lib"        { print
                                 if (g8=="off") print "scf.gemmul8.enable        off       # A/B: plain cuBLAS FP64"
                                 next }
    /<Atoms.SpeciesAndCoordinates/ { print; while ((getline l < coordf) > 0) print l; close(coordf); skip=1; next }
    /Atoms.SpeciesAndCoordinates>/ { print; skip=0; next }
    /<Atoms.UnitVectors/           { print; while ((getline l < cellf)  > 0) print l; close(cellf);  skip=1; next }
    /Atoms.UnitVectors>/           { print; skip=0; next }
    skip { next }
    { print }
  ' "$SRC" > "$c/$c.dat"
  rm -f "$c/.coords.tmp" "$c/.cell.tmp"

  sed -e "s/${TPL}/${c}/g" \
      -e "s/^#PBS -N .*\$/#PBS -N ${jn}/" \
      -e "s/^#PBS -l elapstim_req=.*\$/#PBS -l elapstim_req=${el}/" \
      -e "s/--timeout 1800/--timeout ${tmo}/" \
      -e "s/MPS on, gemmul8 on/MPS on, gemmul8 ${g8}/" \
      -e "s/of 50 max/of ${scfmax} max/" \
      -e "s/^echo \"=== DC-LNO GPU dispatch lines ===\"\$/echo \"=== Band GPU dispatch \/ fallback lines ===\"/" \
      -e "s|^grep -E \"<DC-LNO>\" .*\$|grep -E \"<Band_DFT_Col>\|<Band_DFT_NonCol>\|<Band>\" \"\$STDOUT\" 2>/dev/null \| head -12|" \
      "${TPL}/${TPL}.sh" > "$c/$c.sh"
  rewrite_header "$c/$c.sh" "$c" "$mode" "$atoms" "$kind" "$family" "$m1" "$m2" "$m3" "$scfmax"
  cp "${TPL}/mps_node.sh" "$c/mps_node.sh"

  say "== $c ($family ${m1}x${m2}x${m3}, n=$((13*atoms)), 2n=$((26*atoms)), $el)"
  need "System.Name"   "$c"       "$(awk '$1=="System.Name"{print $2}' "$c/$c.dat")"
  need "Atoms.Number"  "$atoms"   "$(awk '$1=="Atoms.Number"{print $2}' "$c/$c.dat")"
  need "coord lines"   "$atoms"   "$(awk '/<Atoms.SpeciesAndCoordinates/{f=1;next} /Atoms.SpeciesAndCoordinates>/{f=0} f{n++} END{print n+0}' "$c/$c.dat")"
  need "cell lines"    "3"        "$(awk '/<Atoms.UnitVectors/{f=1;next} /Atoms.UnitVectors>/{f=0} f{n++} END{print n+0}' "$c/$c.dat")"
  need "solver"        "Band"     "$(awk '$1=="scf.EigenvalueSolver"{print $2}' "$c/$c.dat")"
  need "Kgrid"         "2 2 2"    "$(awk '$1=="scf.Kgrid"{print $2, $3, $4}' "$c/$c.dat")"
  need "spin"          "$spin"    "$(awk '$1=="scf.SpinPolarization"{print $2}' "$c/$c.dat")"
  need "eigen.lib"     "gpusolver" "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
  need "scf.maxIter"   "$scfmax"  "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
  need "gemmul8 line"  "$([ "$g8" = off ] && echo 1 || echo 0)" "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")"
  need "MD.Type"       "Opt"      "$(awk '$1=="MD.Type"{print $2}' "$c/$c.dat")"
  need "MD.maxIter"    "1"        "$(awk '$1=="MD.maxIter"{print $2}' "$c/$c.dat")"
  need "DATA.PATH"     "../../DFT_DATA19" "$(awk '$1=="DATA.PATH"{print $2}' "$c/$c.dat")"
  need "CASE var"      "CASE=${c}" "$(grep -m1 '^CASE=' "$c/$c.sh")"
  need "NNODE"         "NNODE=1"  "$(grep -m1 '^NNODE=' "$c/$c.sh")"
  need "PPN"           "PPN=48"   "$(grep -m1 '^PPN=' "$c/$c.sh")"
  need "-b nodes"      "#PBS -b 1" "$(grep -m1 '^#PBS -b ' "$c/$c.sh")"
  need "job name"      "#PBS -N ${jn}" "$(grep -m1 '^#PBS -N ' "$c/$c.sh")"
  need "joblog"        "#PBS -o ${c}.joblog" "$(grep -m1 '^#PBS -o ' "$c/$c.sh")"
  need "elapstim"      "#PBS -l elapstim_req=${el}" "$(grep -m1 '^#PBS -l elapstim' "$c/$c.sh")"
  need "cpunum"        "#PBS --cpunum-lhost=48" "$(grep -m1 '^#PBS --cpunum' "$c/$c.sh")"
  need "gpunum"        "1"        "$(grep -c '^#PBS --gpunum-lhost=1' "$c/$c.sh")"
  need "mpirun timeout" "1"       "$(grep -c -- "--timeout ${tmo}" "$c/$c.sh")"
  need "band report"   "1"        "$(grep -cF 'grep -E "<Band_DFT_Col>|<Band_DFT_NonCol>|<Band>"' "$c/$c.sh")"
  need "no stale refs" "0"        "$(grep -c "dia64dc\|DC-LNO\|dc-lno" "$c/$c.sh" "$c/$c.dat" | awk -F: '{s+=$2} END{print s}')"
  need "mps_node.sh"   "yes"      "$([ -s "$c/mps_node.sh" ] && echo yes || echo no)"
  bash -n "$c/$c.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
done

if [ $fail -eq 0 ]; then
  say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"
else
  say "GENERATION HAD FAILURES"; exit 1
fi
