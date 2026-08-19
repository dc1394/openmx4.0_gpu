#!/bin/bash
# CPU best-config preflights (research plan v2.6 sec. 6.2 / step 10):
# 3-SCF elpa2 runs, flat MPI, to (a) verify host-RSS feasibility of the CPU
# path at the Layer-1 sizes and (b) compare 48 vs 36 ranks/node where a
# check is warranted.  Plan rule: flat MPI first; hybrid only if RAM fails.
#
#   sicpu_<s><mode><atoms>_r<ranks>    s: b = band (Kgrid 2 2 2)
#                                         c = cluster (Gamma-only)
# Decks derive from the frozen band decks sib_<mode><atoms>_p with
# scf.eigen.lib gpusolver -> elpa2 and the gemmul8 line dropped (CPU path
# never touches it); cluster cases additionally get the solver/Kgrid edit.
# Job scripts from the proven CPU template dia64dc_1n_c1 (no GPU, no MPS).
#
# These are 1-sample probes: RSS is GO/NO-GO; the 48-vs-36 timing pick
# follows the plan's own 1-3-SCF preflight rule, and only a >10% gap is
# treated as a real preference (single samples on possibly different
# nodes).
#
# Usage from work/:  ./gen_sicpu_probe.sh b:col:300:48 c:nc:216:36 ...
set -u
cd "$(dirname "$0")" || exit 1

TPL=dia64dc_1n_c1
[ -s "$TPL/$TPL.sh" ] || { echo "FATAL: CPU template $TPL missing"; exit 1; }

fail=0; ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

header() {  # $1=case $2=s $3=mode $4=atoms $5=ranks
  local what="band solver, Kgrid 2x2x2 (8 computed k)"
  [ "$2" = c ] && what="cluster solver, Gamma-only"
  cat <<EOF
# CPU best-config preflight (plan v2.6 sec. 6.2 / step 10): $1.
# 3-SCF + force, scf.eigen.lib elpa2, flat MPI: 1 node x $5 ranks (-nt 1),
# whole node reserved (--cpunum-lhost=48), no GPU.  $4 Si atoms, $what.
# Purpose: host-RSS GO/NO-GO for the CPU path at this Layer-1 size, and
# the plan's 1-3-SCF rank-count comparison (48 vs 36) where run.
# Single sample; only a >10% timing gap counts as a rank-count preference.
# Binary: openmx cd5f0d5 build (md5 962f8d2519c2e6aa5a6295513f76fee9).
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
  IFS=: read -r s mode atoms ranks <<<"$pt"
  src="sib_${mode}${atoms}_p"
  [ -s "$src/$src.dat" ] || { say "FAIL missing band deck $src"; fail=1; continue; }
  case "$s" in b|c) : ;; *) say "FAIL bad solver tag $s"; fail=1; continue;; esac
  c="sicpu_${s}${mode}${atoms}_r${ranks}"
  jn="cp${s}${mode:0:1}${atoms}_${ranks}"
  ngen=$((ngen+1)); mkdir -p "$c"

  sed -e "s/^System.Name  *${src}\$/System.Name                     ${c}/" \
      -e "s/^\(scf.eigen.lib  *\)gpusolver\$/\1elpa2/" \
      -e "/^scf.gemmul8.enable/d" \
      "$src/$src.dat" > "$c/$c.dat"
  if [ "$s" = c ]; then
    sed -i -e "s/^\(scf.EigenvalueSolver  *\)Band .*\$/\1cluster     # DC|GDC|Cluster|Band/" \
           -e "s/^\(scf.Kgrid  *\)2 2 2.*\$/\11 1 1       # Gamma-only for the cluster solver/" "$c/$c.dat"
  fi

  # nc band on CPU does 8 dense complex diagonalizations per SCF iteration:
  # give it more wall; everything else fits easily in 40 min.
  el=00:40:00; [ "$s" = b ] && [ "$mode" = nc ] && el=01:10:00
  tmo=$(( $( IFS=:; set -- $el; echo $((10#$1*3600+10#$2*60+10#$3)) ) - 360 ))

  sed -e "s/${TPL}/${c}/g" \
      -e "s/^PPN=48\$/PPN=${ranks}/" \
      -e "s/^#PBS -N .*\$/#PBS -N ${jn}/" \
      -e "s/^#PBS -l elapstim_req=.*\$/#PBS -l elapstim_req=${el}/" \
      -e "s/--timeout [0-9]*/--timeout ${tmo}/" \
      -e "s/of 50 max/of 3 max/" \
      "${TPL}/${TPL}.sh" > "$c/$c.sh"
  rewrite_header "$c/$c.sh" "$c" "$s" "$mode" "$atoms" "$ranks"

  say "== $c ($el)"
  need "System.Name"  "$c"      "$(awk '$1=="System.Name"{print $2}' "$c/$c.dat")"
  need "eigen.lib"    "elpa2"   "$(awk '$1=="scf.eigen.lib"{print $2}' "$c/$c.dat")"
  need "solver"       "$([ "$s" = c ] && echo cluster || echo Band)" "$(awk '$1=="scf.EigenvalueSolver"{print $2}' "$c/$c.dat")"
  need "Kgrid"        "$([ "$s" = c ] && echo '1 1 1' || echo '2 2 2')" "$(awk '$1=="scf.Kgrid"{print $2, $3, $4}' "$c/$c.dat")"
  need "spin"         "$([ "$mode" = nc ] && echo NC || echo off)" "$(awk '$1=="scf.SpinPolarization"{print $2}' "$c/$c.dat")"
  need "no gemmul8"   "0"       "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")"
  need "scf.maxIter"  "3"       "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
  need "atoms"        "$atoms"  "$(awk '$1=="Atoms.Number"{print $2}' "$c/$c.dat")"
  need "CASE var"     "CASE=${c}" "$(grep -m1 '^CASE=' "$c/$c.sh")"
  need "PPN"          "PPN=${ranks}" "$(grep -m1 '^PPN=' "$c/$c.sh")"
  need "cpunum full"  "#PBS --cpunum-lhost=48" "$(grep -m1 '^#PBS --cpunum' "$c/$c.sh")"
  need "no gpunum"    "0"       "$(grep -c '^#PBS --gpunum' "$c/$c.sh")"
  need "no MPS"       "0"       "$(grep -c 'mps_node' "$c/$c.sh")"
  need "elapstim"     "#PBS -l elapstim_req=${el}" "$(grep -m1 '^#PBS -l elapstim' "$c/$c.sh")"
  need "no stale refs" "0"      "$(grep -c "dia64dc\|DC-LNO\|${src}/" "$c/$c.sh")"
  bash -n "$c/$c.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
done

if [ $fail -eq 0 ]; then say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"; else say "GENERATION HAD FAILURES"; exit 1; fi
