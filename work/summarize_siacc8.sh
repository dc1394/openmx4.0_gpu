#!/bin/bash
# H100 practical-convergence rerun aggregate -- the Pegasus twin of
# summarize_s5a8.sh (thesis tab:practical-convergence verification; see
# gen_siacc8.sh).
# The 12 siacc8 series rerun the displaced-Si-216 accuracy decks with the
# practical criterion ITSELF as the input threshold (scf.criterion 1.0e-8,
# scf.maxIter 200).  Two quantities are reported per run and are expected
# to COINCIDE:
#   SCF      = where OpenMX's own convergence logic stopped the run.
#              DFT.c:1330 tests 2<=SCF_iter && dUele<criterion &&
#              NormRD[0]<10*criterion, where NormRD[0] is the SQUARED
#              density residual -- the printed <DFT> NormRD is
#              sqrt(NormRD[0]) (DFT.c:1631) -- so in printed units the
#              second condition reads NormRD < sqrt(10*1e-8) = 3.16e-4:
#              a loose guard whose only practical effect is to reject the
#              SCF-2 dUele=0 mixing artifact.
#   i_conv8  = first arrival at the practical criterion, read off a
#              history: min{ i>=3 : dUele(i) <= 1e-8 }.
# The verdict section then applies the same i_conv8 rule to ALL series of
# both machines: the 24 canonical criterion-1e-13 histories (H100 siacc_*
# and RTX 5080 s5a_*) plus the 24 criterion-1e-8 reruns (RTX 5080 s5a8_*
# and these H100 siacc8_*).
# Run from work/:  ./summarize_siacc8.sh
set -u
cd "$(dirname "$0")" || exit 1

utot()  { grep -E "Utot\." "$1/$1.out" 2>/dev/null | tail -1 | awk '{print $NF}'; }
chemp() { grep -iE "Chemical potential" "$1/$1.out" 2>/dev/null | tail -1 | awk '{print $NF}'; }
nscf()  { grep -cE '^<DFT>  Uele' "$1/$1.std" 2>/dev/null; }
fmax()  { awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{s=$6*$6+$7*$7+$8*$8; if(s>m)m=s} END{if(m>0)printf "%.8f", sqrt(m); else printf "0"}' "$1/$1.out" 2>/dev/null; }
fdiff() {
  paste <(awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{print $6,$7,$8}' "$1/$1.out" 2>/dev/null) \
        <(awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{print $6,$7,$8}' "$2/$2.out" 2>/dev/null) | \
  awk 'NF==6 { for(i=1;i<=3;i++){ d=$i-$(i+3); if(d<0)d=-d; if(d>m)m=d } n++ }
       END { if(n>0) printf "%.3e", m; else printf "n/a" }'
}
ediff() {
  a=$(utot "$1"); b=$(utot "$2")
  [ -n "$a" ] && [ -n "$b" ] && awk -v x="$a" -v y="$b" 'BEGIN{d=x-y; if(d<0)d=-d; printf "%.3e", d}' || echo n/a
}
hist() {  # $1=case -> lines "i dUele NormRD"
  paste <(grep -E '^<DFT>  Uele' "$1/$1.std" 2>/dev/null | awk '{print $NF}') \
        <(grep -E '^<DFT>  NormRD' "$1/$1.std" 2>/dev/null | awk '{print $4}') | \
  awk '{i++; printf "%d %.6e %.6e\n", i, $1, $2}'
}
iconv8() { hist "$1" | awk '$1>=3 && $2+0<=1e-8 {print $1; exit}'; }
stopline() { hist "$1" | tail -1; }
mani() {  # manifest verdict, same gates as summarize_s5a8.sh (cfg c = elpa2)
  python3 - "$1" "$2" <<'EOF'
import json,sys
c,cfg=sys.argv[1],sys.argv[2]
try: d=json.load(open(f"{c}/{c}.manifest.json"))
except Exception: print("NO-MANIFEST"); sys.exit(0)
bad=[]
if d.get("release_tag")!="v2.0_thesis": bad.append("tag")
ds=d["dense_solver"]; g8=d["gemmul8"]
if cfg in ("o","g"):
    if ds["path"]!="gpusolver-gpu-dense": bad.append("path="+ds["path"])
    if ds["cpu_solves"]!=0: bad.append("cpu_solves=%d"%ds["cpu_solves"])
if cfg=="g":
    fb=g8["d_fallbacks"]+g8["z_fallbacks"]
    if fb: bad.append("g8_fb=%d"%fb)
    if g8["d_calls"]+g8["z_calls"]==0: bad.append("g8_nocalls")
if cfg=="o" and (g8["d_calls"]+g8["z_calls"])!=0: bad.append("g8_ran")
if cfg=="c" and d["input"]["eigen_lib"]!="elpa2": bad.append("lib")
print("OK" if not bad else ";".join(bad))
EOF
}

echo "H100 practical-convergence reruns (v2.0_thesis tag build):"
echo "displaced Si 216 (atom 1 +0.05 Ang x), four dense-solver paths,"
echo "scf.criterion 1.0e-8, scf.maxIter 200, np48 nt1."
echo "configs: c = CPU elpa2 | o = GPU cuBLAS + MPS | g = GPU GEMMul8 + MPS"
echo "binary: v2.0_thesis tag build, md5 11227640dc6f8a8b194ddcd9ab811917"
echo "(same binary and decks as the siacc criterion-1e-13 campaign; only"
echo " scf.criterion/scf.maxIter differ)"
echo
echo "=== reruns at criterion 1.0e-8 ==="
printf "%-17s %4s %8s %12s %12s %-22s %-14s %-12s %s\n" case SCF i_conv8 dUele@stop NormRD@stop Utot ChemPot "|F|max" manifest
for s in bcol bnc ccol cnc; do
  for cfg in c o g; do
    c="siacc8_${s}_${cfg}11"
    read -r _ du nr <<<"$(stopline "$c")"
    printf "%-17s %4s %8s %12s %12s %-22s %-14s %-12s %s\n" "$c" "$(nscf "$c")" "$(iconv8 "$c")" "${du:-n/a}" "${nr:-n/a}" "$(utot "$c")" "$(chemp "$c")" "$(fmax "$c")" "$(mani "$c" "$cfg")"
  done
done
echo
echo "--- practical-stop truncation vs the 1e-13 parents (same path+cfg) ---"
printf "  %-6s %-5s %-14s %-14s\n" path cfg "dE(8 vs 13)" "dFmax(8 vs 13)"
for s in bcol bnc ccol cnc; do
  for cfg in c o g; do
    printf "  %-6s %-5s %-14s %-14s\n" "$s" "$cfg" \
      "$(ediff siacc8_${s}_${cfg}11 siacc_${s}_${cfg}11)" "$(fdiff siacc8_${s}_${cfg}11 siacc_${s}_${cfg}11)"
  done
done
echo
echo "--- cross-config differences at criterion 1e-8 (Ha / Ha per Bohr) ---"
printf "  %-6s %-14s %-14s %-14s %-14s\n" path "dE(CPU-cuB)" "dFmax(CPU-cuB)" "dE(G8-cuB)" "dFmax(G8-cuB)"
for s in bcol bnc ccol cnc; do
  printf "  %-6s %-14s %-14s %-14s %-14s\n" "$s" \
    "$(ediff siacc8_${s}_c11 siacc8_${s}_o11)" "$(fdiff siacc8_${s}_c11 siacc8_${s}_o11)" \
    "$(ediff siacc8_${s}_g11 siacc8_${s}_o11)" "$(fdiff siacc8_${s}_g11 siacc8_${s}_o11)"
done
echo
echo "=== i_conv8 post hoc over the canonical criterion-1e-13 histories ==="
echo "(rule: min{ i>=3 : dUele(i) <= 1e-8 }; SCF-2 artifact excluded)"
printf "%-6s | %-22s | %-22s\n" "" "H100 (siacc_*)" "RTX 5080 (s5a_*)"
printf "%-6s | %6s %6s %7s | %6s %6s %7s\n" path CPU cuBLAS GEMMul8 CPU cuBLAS GEMMul8
for s in bcol bnc ccol cnc; do
  printf "%-6s | %6s %6s %7s | %6s %6s %7s\n" "$s" \
    "$(iconv8 siacc_${s}_c11)" "$(iconv8 siacc_${s}_o11)" "$(iconv8 siacc_${s}_g11)" \
    "$(iconv8 s5a_${s}_c1611)" "$(iconv8 s5a_${s}_o11)" "$(iconv8 s5a_${s}_g11)"
done
echo
echo "=== verdict vs thesis tab:practical-convergence (bcol 23, bnc 23, ccol 20, cnc 23) ==="
declare -A want=([bcol]=23 [bnc]=23 [ccol]=20 [cnc]=23)
allok=1
for s in bcol bnc ccol cnc; do
  vals=""
  for c in siacc_${s}_c11 siacc_${s}_o11 siacc_${s}_g11 s5a_${s}_c1611 s5a_${s}_o11 s5a_${s}_g11 \
           s5a8_${s}_c1611 s5a8_${s}_o11 s5a8_${s}_g11 siacc8_${s}_c11 siacc8_${s}_o11 siacc8_${s}_g11; do
    v=$(iconv8 "$c"); vals="$vals ${v:-?}"
    [ "${v:-x}" = "${want[$s]}" ] || allok=0
  done
  printf "  %-5s want %2s got:%s\n" "$s" "${want[$s]}" "$vals"
done
if [ "$allok" -eq 1 ]; then
  echo "  ALL 48 SERIES MATCH (24 canonical 1e-13 histories + 24 criterion-1e-8 reruns, both GPUs)"
else
  echo "  MISMATCH PRESENT -- see rows above"
fi
