#!/bin/bash
# RTX 5080 accuracy aggregate (rtx5080_procedure.md R4; plan v2.6 sec.
# 7.5-1 / 13.2-B / 13.3) -- displaced-Si-216 port of summarize_siacc_thesis.sh
# (the MnO / Cr2 / GaAs legs are Pegasus-side and are not re-run here):
#   s5a_{bcol,bnc,ccol,cnc}_{c16,o,g}11 (+ bnc o12/g12)  displaced Si 216
# Targets (13.3): |dE| <= 1e-10 Ha, |dF|max <= 1e-6 Ha/Bohr, judged
# against the run-to-run baseline (o11-vs-o12, g11-vs-g12).
# Run from work/:  ./summarize_s5a.sh
set -u
cd "$(dirname "$0")" || exit 1

utot()  { grep -E "Utot\." "$1/$1.out" 2>/dev/null | tail -1 | awk '{print $NF}'; }
chemp() { grep -iE "Chemical potential" "$1/$1.out" 2>/dev/null | tail -1 | awk '{print $NF}'; }
nscf()  { grep -m1 "SCF iterations completed" -A1 "$1/$1.joblog" 2>/dev/null | tail -1; }
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
mani() {  # manifest verdict, same gates as summarize_s5p.sh
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
if cfg=="c16" and d["input"]["eigen_lib"]!="elpa2": bad.append("lib")
print("OK" if not bad else ";".join(bad))
EOF
}

echo "RTX 5080 accuracy campaign (v2.0_thesis tag build): displaced Si 216"
echo "(atom 1 +0.05 Ang x; forces are non-zero), four dense-solver paths,"
echo "run to convergence (criterion 1e-13, up to 60 SCF), np16 nt1."
echo "configs: c16 = CPU elpa2-16 | o = GPU cuBLAS + MPS | g = GPU GEMMul8 + MPS"
echo "targets (plan 13.3): |dE| <= 1e-10 Ha, |dF|max <= 1e-6 Ha/Bohr"
echo "binary: v2.0_thesis tag build, md5 e3e64c35f48a9058cd98b5e923810ae3"
echo
echo "=== displaced Si, 216 atoms ==="
printf "%-16s %6s %-24s %-14s %-13s %s\n" case SCF Utot ChemPot "|F|max" manifest
for s in bcol bnc ccol cnc; do
  for cfg in c16 o g; do
    c="s5a_${s}_${cfg}11"
    printf "%-16s %6s %-24s %-14s %-13s %s\n" "$c" "$(nscf "$c")" "$(utot "$c")" "$(chemp "$c")" "$(fmax "$c")" "$(mani "$c" "$cfg")"
  done
done
for cfg in o g; do
  c="s5a_bnc_${cfg}12"
  printf "%-16s %6s %-24s %-14s %-13s %s\n" "$c" "$(nscf "$c")" "$(utot "$c")" "$(chemp "$c")" "$(fmax "$c")" "$(mani "$c" "$cfg")"
done
echo
echo "--- cross-config differences per solver path (Ha / Ha per Bohr) ---"
printf "  %-6s %-14s %-14s %-14s %-14s\n" path "dE(CPU-cuB)" "dFmax(CPU-cuB)" "dE(G8-cuB)" "dFmax(G8-cuB)"
for s in bcol bnc ccol cnc; do
  printf "  %-6s %-14s %-14s %-14s %-14s\n" "$s" \
    "$(ediff s5a_${s}_c1611 s5a_${s}_o11)" "$(fdiff s5a_${s}_c1611 s5a_${s}_o11)" \
    "$(ediff s5a_${s}_g11 s5a_${s}_o11)" "$(fdiff s5a_${s}_g11 s5a_${s}_o11)"
done
echo
echo "--- run-to-run baseline (bnc, same config twice) ---"
printf "  cuBLAS o11-o12:  dE=%s  dFmax=%s\n" "$(ediff s5a_bnc_o11 s5a_bnc_o12)" "$(fdiff s5a_bnc_o11 s5a_bnc_o12)"
printf "  GEMMul8 g11-g12: dE=%s  dFmax=%s\n" "$(ediff s5a_bnc_g11 s5a_bnc_g12)" "$(fdiff s5a_bnc_g11 s5a_bnc_g12)"
echo
echo "=== GPU dispatch evidence (all four displaced-Si GPU runs must engage) ==="
for s in bcol bnc ccol cnc; do
  c="s5a_${s}_o11"
  ev=$(grep -m1 -oE "(GPU device [0-9]+: [0-9]+ k-owner rank[^,]*|<Set_Hamiltonian> GPU device[^,]*)" "$c/$c.joblog" 2>/dev/null)
  fb=$(grep -ciE "cannot fit|Falling back" "$c/$c.joblog" 2>/dev/null)
  printf "  %-16s [%s] fallback-lines=%s\n" "$c" "${ev:-no GPU banner}" "$fb"
done
