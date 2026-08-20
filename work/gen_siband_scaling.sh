#!/bin/bash
# Si diamond BAND limited strong scaling (research plan v2.6 sec. 7.4 / 8.5):
# generate 1/2/4/8-node cases at the mode representatives fixed by the
# sec.-7.3 size search, 48 ranks/node, -nt 1, MPS on, reps per point plus one
# 3-SCF preflight per (mode, node count).  8 nodes = 1 k-owner per GPU, the
# natural endpoint of the 8-computed-k parallelism (thesis extension).
#
#   sibs_<mode><atoms>_<N>n_<b><r>   b: o = gemmul8 off (cuBLAS FP64)
#                                       g = gemmul8 on  (INT8 emulation)
#                                       w = 3-SCF preflight (gemmul8 on),
#                                           warm-up + path check, 1 per
#                                           (mode, node count), NOT a rep
#
# The .dat is copied byte-for-byte from the CONFIRMED search case
# (sib_<mode><atoms>_{o,g}) so the scaling series uses exactly the input the
# 25-SCF confirmation validated, with two sed edits:
#   - System.Name
#   - scf.criterion 1e-13 -> 1e-15: nc250_o converged at SCF=24, so 1e-13
#     cannot pin the iteration count; 1e-15 is never reached and every run
#     does exactly scf.maxIter iterations (plan 9.1 "fixed 25 SCF").
#     Documented deviation from ../sidia.dat.
#
# Usage from work/:  ./gen_siband_scaling.sh col:300 nc:216 [...]
#   optionally filter points:  ./gen_siband_scaling.sh col:300 only:2:g:1
#   (only:<N>:<b>:<r> limits generation to matching points; repeatable)
set -u
cd "$(dirname "$0")" || exit 1

fail=0; ngen=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

MODES=""; ONLY=""
for a in "$@"; do
  case "$a" in
    only:*) ONLY="$ONLY ${a#only:}";;
    col:*|nc:*) MODES="$MODES $a";;
    *) say "FAIL bad arg $a"; exit 1;;
  esac
done
[ -n "$MODES" ] || { say "usage: $0 col:<atoms> nc:<atoms> [only:<N>:<b>:<r>]"; exit 1; }

wantpt() {  # $1=N $2=b $3=r
  [ -z "$ONLY" ] && return 0
  case " $ONLY " in *" $1:$2:$3 "*) return 0;; esac
  return 1
}

elap_for() {  # $1=mode $2=b  (single 25-SCF run took col 183 s / nc 299 s at
              #  1 node; multi-node only gets faster; 40/60 min is >8x slack)
  case "$1:$2" in
    col:w|nc:w) echo 00:30:00;;
    col:*)      echo 00:40:00;;
    nc:*)       echo 01:00:00;;
  esac
}
secs() { IFS=: read -r h m s <<<"$1"; echo $((10#$h*3600 + 10#$m*60 + 10#$s)); }

header() {  # $1=case $2=mode $3=atoms $4=N $5=b $6=scfmax
  local what reps
  case "$5" in
    o) what="gemmul8 OFF -> plain cuBLAS FP64 (production rep)";;
    g) what="gemmul8 on, INT8-based FP64 emulation (production rep)";;
    w) what="3-SCF preflight: warm-up + path check, NOT pooled into means";;
  esac
  cat <<EOF
# Si diamond BAND limited strong scaling (plan v2.6 sec. 7.4/8.5): $1.
# ${4} node(s) x 48 ranks (-nt 1), 1 H100/node, MPS on; same input at every
# node count (strong scaling, k-point parallelism over the 8 computed
# k points: 8/4/2/1 k per GPU at 1/2/4/8 nodes).  Config: ${what}.
# System: $3 Si atoms from the sec.-7.3 size search ($2 band representative);
# deck copied from the confirmed sib_$2$3 case, with scf.criterion 1e-15 so
# every run does exactly ${6} SCF iterations (1e-13 converged at 24 once).
# Binary: work/openmx as deployed at submission; the authoritative build
# identity (commit/tag/md5) is recorded in <case>.manifest.json.
# 3 reps per production point; reps landing on bnode013/bnode033 are
# discarded and re-measured under a new rep number.  Wall time = Max_Time
# (slowest rank).  A crash here is a result, not a harness failure.
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

for mspec in $MODES; do
  IFS=: read -r mode atoms <<<"$mspec"
  for b in w o g; do
    # deck source: confirmed 25-SCF case; w reuses the g deck cut to 3 SCF
    case "$b" in
      o) src="sib_${mode}${atoms}_o";;
      g|w) src="sib_${mode}${atoms}_g";;
    esac
    [ -s "$src/$src.dat" ] || { say "FAIL missing confirmed deck $src"; fail=1; continue; }
    tpl="$src"
    for N in 1 2 4 8; do
      case "$b" in w) rlist="0";; *) rlist="1 2 3 4 5 6 11 12 13 14 15 16 17 18 19 20";; esac  # 11+ = thesis band (v2.0_thesis re-measurement)
      for r in $rlist; do
        [ "$b" != w ] && [ -z "$ONLY" ] && [ "$r" -gt 3 ] && continue
        wantpt "$N" "$b" "$r" || continue
        if [ "$b" = w ]; then c="sibs_${mode}${atoms}_${N}n_w"; jn="s${mode:0:1}${atoms}_${N}w"
        else c="sibs_${mode}${atoms}_${N}n_${b}${r}"; jn="s${mode:0:1}${atoms}_${N}${b}${r}"; fi
        el=$(elap_for "$mode" "$b"); tmo=$(( $(secs "$el") - 360 ))
        scfmax=25; [ "$b" = w ] && scfmax=3
        ngen=$((ngen+1)); mkdir -p "$c"

        sed -e "s/^System.Name  *${src}\$/System.Name                     ${c}/" \
            -e "s/^scf.criterion  *1.0e-13.*\$/scf.criterion             1.0e-15      # pin the SCF count (see header)/" \
            -e "s/^scf.maxIter  *25.*\$/scf.maxIter                 ${scfmax}          # fixed-iteration timing/" \
            "$src/$src.dat" > "$c/$c.dat"

        sed -e "s/${tpl}/${c}/g" \
            -e "s/^NNODE=1\$/NNODE=${N}/" \
            -e "s/^#PBS -b 1\$/#PBS -b ${N}/" \
            -e "s/^#PBS -N .*\$/#PBS -N ${jn}/" \
            -e "s/^#PBS -l elapstim_req=.*\$/#PBS -l elapstim_req=${el}/" \
            -e "s/--timeout [0-9]*/--timeout ${tmo}/" \
            -e "s/of 25 max/of ${scfmax} max/" \
            "${tpl}/${tpl}.sh" > "$c/$c.sh"
        rewrite_header "$c/$c.sh" "$c" "$mode" "$atoms" "$N" "$b" "$scfmax"
        cp "${tpl}/mps_node.sh" "$c/mps_node.sh"

        say "== $c ($el)"
        need "System.Name"  "$c"          "$(awk '$1=="System.Name"{print $2}' "$c/$c.dat")"
        need "criterion"    "1.0e-15"     "$(awk '$1=="scf.criterion"{print $2}' "$c/$c.dat")"
        need "scf.maxIter"  "$scfmax"     "$(awk '$1=="scf.maxIter"{print $2}' "$c/$c.dat")"
        need "atoms"        "$atoms"      "$(awk '$1=="Atoms.Number"{print $2}' "$c/$c.dat")"
        need "gemmul8 line" "$([ "$b" = o ] && echo 1 || echo 0)" "$(grep -c '^scf.gemmul8.enable' "$c/$c.dat")"
        need "spin"         "$([ "$mode" = nc ] && echo NC || echo off)" "$(awk '$1=="scf.SpinPolarization"{print $2}' "$c/$c.dat")"
        need "coord lines"  "$atoms"      "$(awk '/<Atoms.SpeciesAndCoordinates/{f=1;next} /Atoms.SpeciesAndCoordinates>/{f=0} f{n++} END{print n+0}' "$c/$c.dat")"
        need "CASE var"     "CASE=${c}"   "$(grep -m1 '^CASE=' "$c/$c.sh")"
        need "NNODE"        "NNODE=${N}"  "$(grep -m1 '^NNODE=' "$c/$c.sh")"
        need "PPN"          "PPN=48"      "$(grep -m1 '^PPN=' "$c/$c.sh")"
        need "-b nodes"     "#PBS -b ${N}" "$(grep -m1 '^#PBS -b ' "$c/$c.sh")"
        need "gpunum"       "1"           "$(grep -c '^#PBS --gpunum-lhost=1' "$c/$c.sh")"
        need "elapstim"     "#PBS -l elapstim_req=${el}" "$(grep -m1 '^#PBS -l elapstim' "$c/$c.sh")"
        need "no stale src" "0"           "$(grep -c "sib_${mode}${atoms}_[og]/" "$c/$c.sh")"
        bash -n "$c/$c.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
      done
    done
  done
done

if [ $fail -eq 0 ]; then say "ALL ${ngen} CASE(S) GENERATED AND ASSERTED OK"; else say "GENERATION HAD FAILURES"; exit 1; fi
