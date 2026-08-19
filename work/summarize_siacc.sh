#!/bin/bash
# Aggregate the accuracy campaign (plan v2.6 sec. 7.5 / 13.2-B / 13.3):
#   siacc_{bcol,bnc,ccol,cnc}_{c,o,g}1 (+ bnc o2/g2)  displaced Si 216
#   siacc_mno_{c,o,g}1                                collinear AFM MnO
#   siacc_{cr2,gaas}_{c,o,g}1                         NC+SOC (tiny)
# Targets (13.3): |dE| <= 1e-10 Ha, |dF|max <= 1e-6 Ha/Bohr, judged
# against the run-to-run baseline (o1-vs-o2, g1-vs-g2).
# Run from work/:  ./summarize_siacc.sh
set -u
cd "$(dirname "$0")" || exit 1

utot()  { grep -E "Utot\." "$1/$1.out" 2>/dev/null | tail -1 | awk '{print $NF}'; }
chemp() { grep -iE "Chemical potential" "$1/$1.out" 2>/dev/null | tail -1 | awk '{print $NF}'; }
nscf()  { grep -m1 "SCF iterations completed" -A1 "$1/$1.joblog" 2>/dev/null | tail -1; }
conv()  { grep -qE "SCF history .*converged|The SCF was achieved" "$1/$1.std" 2>/dev/null && echo yes || \
          { n=$(nscf "$1"); m=$(awk '$1=="scf.maxIter"{print $2}' "$1/$1.dat"); [ -n "$n" ] && [ "$n" -lt "${m:-0}" ] && echo "yes(iter<max)" || echo "CHECK"; }; }
fmax()  { awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{s=$6*$6+$7*$7+$8*$8; if(s>m)m=s} END{if(m>0)printf "%.8f", sqrt(m); else printf "0"}' "$1/$1.out" 2>/dev/null; }
# max abs per-component force difference between two runs
fdiff() {
  paste <(awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{print $6,$7,$8}' "$1/$1.out" 2>/dev/null) \
        <(awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{print $6,$7,$8}' "$2/$2.out" 2>/dev/null) | \
  awk 'NF==6 { for(i=1;i<=3;i++){ d=$i-$(i+3); if(d<0)d=-d; if(d>m)m=d } n++ }
       END { if(n>0) printf "%.3e", m; else printf "n/a" }'
}
ediff() {  # |Utot(a)-Utot(b)|
  a=$(utot "$1"); b=$(utot "$2")
  [ -n "$a" ] && [ -n "$b" ] && awk -v x="$a" -v y="$b" 'BEGIN{d=x-y; if(d<0)d=-d; printf "%.3e", d}' || echo n/a
}
moment() { grep -iE "Total spin moment" "$1/$1.out" 2>/dev/null | tail -1 | sed 's/^ *//'; }
# per-atom Mulliken rows (first N atoms) from the "Up Down Sum Diff [theta phi]"
# table -- for AFM systems the TOTAL moment is 0 by construction and the
# sublattice (per-atom) moment is the discriminating quantity; for NC the
# theta/phi columns carry the >=2 non-zero components required by plan 7.5-3.
atmom() {  # $1=case $2=natoms
  awk -v n="$2" '/^ *Up( spin)? +Down( spin)? +Sum +Diff/{f=1; next} f && NF>=6 && $1+0>0 { print "                   " $0; if (++k>=n) exit }' "$1/$1.out" 2>/dev/null
}

echo "Accuracy campaign: displaced Si 216 (4 solver paths), Crys-MnO"
echo "(collinear AFM), Cr2 and GaAs (NC+SOC, tiny -> below GPU dispatch"
echo "thresholds: correctness check, not GPU numerics)."
echo "configs: c = CPU elpa2-48 | o = GPU cuBLAS + MPS | g = GPU GEMMul8 + MPS"
echo "targets (plan 13.3): |dE| <= 1e-10 Ha, |dF|max <= 1e-6 Ha/Bohr"
echo "binary: cd5f0d5 + GEMMul8 v3.2.0, md5 962f8d2519c2e6aa5a6295513f76fee9"
echo
echo "=== displaced Si, 216 atoms (atom 1 +0.05 Ang x; forces are non-zero) ==="
printf "%-16s %6s %-24s %-14s %-12s\n" case SCF Utot ChemPot "|F|max"
for s in bcol bnc ccol cnc; do
  for cfg in c o g; do
    c="siacc_${s}_${cfg}1"
    printf "%-16s %6s %-24s %-14s %-12s\n" "$c" "$(nscf "$c")" "$(utot "$c")" "$(chemp "$c")" "$(fmax "$c")"
  done
done
for c in siacc_bnc_o2 siacc_bnc_g2; do
  printf "%-16s %6s %-24s %-14s %-12s\n" "$c" "$(nscf "$c")" "$(utot "$c")" "$(chemp "$c")" "$(fmax "$c")"
done
echo
echo "--- cross-config differences per solver path (Ha / Ha per Bohr) ---"
printf "  %-6s %-14s %-14s %-14s %-14s\n" path "dE(CPU-cuB)" "dFmax(CPU-cuB)" "dE(G8-cuB)" "dFmax(G8-cuB)"
for s in bcol bnc ccol cnc; do
  printf "  %-6s %-14s %-14s %-14s %-14s\n" "$s" \
    "$(ediff siacc_${s}_c1 siacc_${s}_o1)" "$(fdiff siacc_${s}_c1 siacc_${s}_o1)" \
    "$(ediff siacc_${s}_g1 siacc_${s}_o1)" "$(fdiff siacc_${s}_g1 siacc_${s}_o1)"
done
echo
echo "--- run-to-run baseline (bnc, same config twice) ---"
printf "  cuBLAS o1-o2:  dE=%s  dFmax=%s\n" "$(ediff siacc_bnc_o1 siacc_bnc_o2)" "$(fdiff siacc_bnc_o1 siacc_bnc_o2)"
printf "  GEMMul8 g1-g2: dE=%s  dFmax=%s\n" "$(ediff siacc_bnc_g1 siacc_bnc_g2)" "$(fdiff siacc_bnc_g1 siacc_bnc_g2)"
echo
echo "=== Crys-MnO (collinear AFM, band 5x5x5; plan 7.5-2) ==="
for cfg in c o g; do
  c="siacc_mno_${cfg}1"
  printf "%-16s SCF=%-4s Utot=%-22s |F|max=%-12s\n" "$c" "$(nscf "$c")" "$(utot "$c")" "$(fmax "$c")"
  printf "                 %s\n" "$(moment "$c")"
  atmom "$c" 2
done
printf "  dE(CPU-cuB)=%s dFmax=%s | dE(G8-cuB)=%s dFmax=%s\n" \
  "$(ediff siacc_mno_c1 siacc_mno_o1)" "$(fdiff siacc_mno_c1 siacc_mno_o1)" \
  "$(ediff siacc_mno_g1 siacc_mno_o1)" "$(fdiff siacc_mno_g1 siacc_mno_o1)"
echo "  (AFM: the TOTAL moment is 0 by construction; the per-atom rows above"
echo "   carry the sublattice moments that discriminate the configs)"
echo
echo "=== NC + SOC (plan 7.5-3; tiny systems, correctness check) ==="
for s in cr2 gaas; do
  for cfg in c o g; do
    c="siacc_${s}_${cfg}1"
    printf "%-16s SCF=%-4s Utot=%-22s |F|max=%-12s\n" "$c" "$(nscf "$c")" "$(utot "$c")" "$(fmax "$c")"
    printf "                 %s\n" "$(moment "$c")"
    atmom "$c" 2
  done
  printf "  dE(CPU-cuB)=%s dFmax=%s | dE(G8-cuB)=%s dFmax=%s\n" \
    "$(ediff siacc_${s}_c1 siacc_${s}_o1)" "$(fdiff siacc_${s}_c1 siacc_${s}_o1)" \
    "$(ediff siacc_${s}_g1 siacc_${s}_o1)" "$(fdiff siacc_${s}_g1 siacc_${s}_o1)"
done
echo
echo "=== GPU dispatch evidence (displaced-Si GPU runs must engage; tiny"
echo "    systems are EXPECTED to stay below the dispatch threshold) ==="
for c in siacc_bcol_o1 siacc_bnc_o1 siacc_ccol_o1 siacc_cnc_o1 siacc_mno_o1 siacc_cr2_o1 siacc_gaas_o1; do
  ev=$(grep -m1 -oE "(GPU device [0-9]+: [0-9]+ k-owner rank[^,]*|<Set_Hamiltonian> GPU device[^,]*)" "$c/$c.joblog" 2>/dev/null)
  fb=$(grep -ciE "cannot fit|Falling back" "$c/$c.joblog" 2>/dev/null)
  printf "  %-16s [%s] fallback-lines=%s\n" "$c" "${ev:-no GPU banner}" "$fb"
done
echo
echo "--- node landings (slow: bnode013, bnode033; accuracy values are"
echo "    node-independent, listed for completeness) ---"
for d in siacc_*; do
  [ -s "$d/$d.env" ] || continue
  nl=$(awk '/^=== nodes ===/{f=1;next} /^=== /{f=0} f&&NF==1{printf "%s ",$1}' "$d/$d.env")
  printf "  %-16s %s\n" "$d" "$nl"
done
