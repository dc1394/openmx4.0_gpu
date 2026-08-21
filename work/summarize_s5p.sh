#!/bin/bash
# RTX 5080 production matrix aggregate (rtx5080_procedure.md R2/R3; plan
# v2.6 sec. 7.2/8.2/8.3) -- port of summarize_siprod_thesis.sh:
#   Cases: s5p_<combo>_<cfg><rep>, thesis rep band:
#     c16: 11-13   o: 11-15   g: 11-15   om/gm (ablation combos): 11-13
# Layer 3: S_G8 = T(o)/T(g).  CPU ref (np16, reference only per plan 8.2):
# T(c16)/T(o).  MPS: T(om)/T(o) and, on the gm combo, T(gm)/T(g).
# Ratio errors: relative SDs in quadrature; error bar crossing 1 -> "n.s.".
# The manifest of every run is checked against plan sec. 10.2.
# Run from work/:  ./summarize_s5p.sh
set -u
cd "$(dirname "$0")" || exit 1

COMBOS="bnc64 bnc128 bnc216 cnc64 cnc128 cnc216 bcol216 bcol384 ccol216 ccol512"

reps() {  # $1=combo $2=cfg
  case "$2" in
    c16|om|gm) echo "11 12 13";;
    o|g)       echo "11 12 13 14 15";;
  esac
}
has_c16() { case "$1" in bnc216|cnc216|bcol216|ccol216) return 0;; *) return 1;; esac; }
has_om()  { case "$1" in bnc216|cnc216) return 0;; *) return 1;; esac; }
has_gm()  { case "$1" in cnc216) return 0;; *) return 1;; esac; }

collect() {
  for c in $1; do
    line=$(grep -hm1 "^TIMING ${c} " "${c}/${c}.joblog" 2>/dev/null)
    if [ -z "$line" ]; then echo "${c} - - - - MISSING"
    elif echo "$line" | grep -q INCOMPLETE; then echo "${c} - - - - INCOMPLETE"
    else echo "$line" | sed -E 's/^TIMING ([^ ]+) total=([^ ]+) dft=([^ ]+) diag=([^ ]+) set_hamiltonian=([^ ]+)/\1 \2 \3 \4 \5 OK/'
    fi
  done
}

# manifest path-quality verdict (plan 10.2) for one run dir
manifest_check() {  # $1=case $2=cfg
  python3 - "$1" "$2" <<'EOF'
import json,sys
c,cfg=sys.argv[1],sys.argv[2]
try:
    d=json.load(open(f"{c}/{c}.manifest.json"))
except Exception:
    print("NO-MANIFEST"); sys.exit(0)
bad=[]
if d.get("release_tag")!="v2.0_thesis": bad.append("tag="+str(d.get("release_tag")))
ds=d["dense_solver"]; g8=d["gemmul8"]; prof=d.get("profiling",{})
if prof.get("mode")!="production" or prof.get("external_profiler"): bad.append("profiled")
if cfg in ("o","g","om","gm"):
    if ds["path"]!="gpusolver-gpu-dense": bad.append("path="+ds["path"])
    if ds["cpu_solves"]!=0: bad.append("cpu_solves=%d"%ds["cpu_solves"])
if cfg in ("g","gm"):
    fb=g8["d_fallbacks"]+g8["z_fallbacks"]
    if fb: bad.append("g8_fb=%d"%fb)
    if not g8["enabled"]: bad.append("g8_off")
    if g8["d_calls"]+g8["z_calls"]==0: bad.append("g8_nocalls")
if cfg in ("o","om") and (g8["d_calls"]+g8["z_calls"])!=0: bad.append("g8_ran")
if cfg=="c16" and d["input"]["eigen_lib"]!="elpa2": bad.append("lib="+d["input"]["eigen_lib"])
mps=d["layout"]["mps"]["detected"]
if cfg in ("o","g") and not mps: bad.append("no-mps")
# om/gm: the manifest's detected flag keys on the exported pipe-dir env var,
# which the MPS-off jobs still carry without ever starting a daemon; the
# authoritative evidence is the job's own control_pipe=no assertion, checked
# by the shell caller below (MPSOFF-EVIDENCE).
print("OK" if not bad else ";".join(bad))
EOF
}

echo "RTX 5080 production matrix (v2.0_thesis tag build): Layer 3 (cuBLAS vs"
echo "GEMMul8), MPS ablation, CPU np16 reference.  16 ranks (-nt 1), 1 box"
echo "(Core i9-10980XE, RTX 5080 16 GB, MPS on for o/g).  Fixed 25 SCF"
echo "(criterion 1e-15), Max_Time column (s).  Path quality: every run's"
echo "manifest must satisfy plan sec. 10.2 (right column all OK; the bnc216 g"
echo "reserve_policy fallbacks are a pre-declared formal result of the 16 GB"
echo "card -- see rtx5080_procedure.md R1)."
echo
echo "--- per-run (with manifest verdict) ---"
printf "%-20s %10s %10s %10s %10s  %-6s %s\n" case total DFT Diag Set_Ham status manifest
for combo in $COMBOS; do
  for cfg in c16 o g om gm; do
    [ "$cfg" = c16 ] && ! has_c16 "$combo" && continue
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    [ "$cfg" = gm ] && ! has_gm "$combo" && continue
    for r in $(reps "$combo" "$cfg"); do
      c="s5p_${combo}_${cfg}${r}"
      collect "$c" | awk '{printf "%-20s %10s %10s %10s %10s  %-6s", $1,$2,$3,$4,$5,$6}'
      v=$(manifest_check "$c" "$cfg")
      case "$cfg" in
        om|gm)
          if grep -q "control_pipe=no" "$c/$c.env" 2>/dev/null; then v="$v mps-off-verified"
          else v="$v MPSOFF-UNPROVEN"; fi;;
      esac
      printf " %s\n" "$v"
    done
  done
done
echo

echo "--- statistics (mean +- sample SD; CV flag at >2%) ---"
: > /tmp/.s5p_means
for combo in $COMBOS; do
  echo "  [$combo]"
  for cfg in c16 o g om gm; do
    [ "$cfg" = c16 ] && ! has_c16 "$combo" && continue
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    [ "$cfg" = gm ] && ! has_gm "$combo" && continue
    cl=""; for r in $(reps "$combo" "$cfg"); do cl="$cl s5p_${combo}_${cfg}${r}"; done
    collect "$cl" | awk -v combo="$combo" -v cfg="$cfg" '
      $6=="OK" { n++; for(i=2;i<=5;i++){s[i]+=$i;q[i]+=$i*$i} }
      $6!="OK" { bad=bad" "$1 }
      END {
        if (n==0) { printf "    %-4s no usable runs:%s\n", cfg, bad; exit }
        split("2 4 5", idx," "); split("Total Diag Set_Ham", nm," ")
        printf "    %-4s (n=%d)%s", cfg, n, (bad==""?"":" [excl:"bad"]")
        for (k=1;k<=3;k++) {
          i=idx[k]; m=s[i]/n; sd=(n>1)?sqrt((q[i]-n*m*m)/(n-1)):0
          cv=(m>0)?100*sd/m:0
          printf "  %s %9.2f+-%5.2f%s", nm[k], m, sd, (k==1 && cv>2 ? " (CV " sprintf("%.1f",cv) "%% **>2%%**)" : "")
          printf "MEAN %s %s %s %.4f %.4f\n", combo, cfg, nm[k], m, sd > "/dev/stderr"
        }
        printf "\n"
      }' 2>>/tmp/.s5p_means
  done
done
echo

echo "--- Layer 3 / CPU-ref / MPS ratios (Total and Diag; +- in quadrature) ---"
awk '
  $1=="MEAN" { m[$2" "$3" "$4]=$5; sd[$2" "$3" "$4]=$6 }
  function ratio(combo, a, b, label, ph,   ka, kb, r, rel, sig) {
    ka=combo" "a" "ph; kb=combo" "b" "ph
    if (!((ka in m)&&(kb in m)) || m[kb]<=0) return
    r=m[ka]/m[kb]; rel=sqrt((sd[ka]/m[ka])^2+(sd[kb]/m[kb])^2)
    sig=(r-1>r*rel || 1-r>r*rel) ? "" : "  [not significant]"
    printf "    %-30s %-6s %6.3f +- %.3f%s\n", label, ph, r, r*rel, sig
  }
  END {
    split("bnc64 bnc128 bnc216 cnc64 cnc128 cnc216 bcol216 bcol384 ccol216 ccol512", C," ")
    for (i=1;i<=10;i++) {
      printf "  [%s]\n", C[i]
      ratio(C[i], "c16", "o", "CPUref(np16)/GPU speedup", "Total")
      ratio(C[i], "c16", "o", "CPUref(np16)/GPU speedup", "Diag")
      ratio(C[i], "g", "o", "Layer3 G8/cuBLAS (>1 slower)", "Total")
      ratio(C[i], "g", "o", "Layer3 G8/cuBLAS (>1 slower)", "Diag")
      ratio(C[i], "om", "o", "MPSoff/MPSon (>1 = MPS helps)", "Total")
      ratio(C[i], "om", "o", "MPSoff/MPSon (>1 = MPS helps)", "Diag")
      ratio(C[i], "gm", "g", "G8: MPSoff/MPSon (>1 = helps)", "Total")
      ratio(C[i], "gm", "g", "G8: MPSoff/MPSon (>1 = helps)", "Diag")
    }
  }' /tmp/.s5p_means
rm -f /tmp/.s5p_means
echo

echo "--- Utot consistency (per combo: each cfg's last-rep value) ---"
for combo in $COMBOS; do
  printf "  %-8s" "$combo"
  for cfg in c16 o g om gm; do
    [ "$cfg" = c16 ] && ! has_c16 "$combo" && continue
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    [ "$cfg" = gm ] && ! has_gm "$combo" && continue
    r=$(reps "$combo" "$cfg" | awk '{print $NF}')
    u=$(grep -E "Utot\." "s5p_${combo}_${cfg}${r}/s5p_${combo}_${cfg}${r}.out" 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "  %s=%s" "$cfg" "${u:-?}"
  done
  printf "\n"
done
echo

echo "--- thermal-flag audit (procedure 0-6: nonzero => discard + re-rep) ---"
grep -E "^\[.*s5p_" s5_queue.progress 2>/dev/null | awk '$4!="thermal_flags=0" && $3!~/rc=/ {print}  $0~/thermal_flags=[1-9]/ {print "  " $0}' | sort -u
echo "  (no lines above = no thermal slowdown in any rep)"
echo

echo "--- peak memory (manifest: device VRAM used / min free; rank VmHWM) ---"
for combo in $COMBOS; do
  for cfg in o g om gm; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    [ "$cfg" = gm ] && ! has_gm "$combo" && continue
    r=$(reps "$combo" "$cfg" | awk '{print $1}')
    c="s5p_${combo}_${cfg}${r}"
    python3 - "$c" <<'EOF' 2>/dev/null
import json,sys
c=sys.argv[1]
try:
    d=json.load(open(f"{c}/{c}.manifest.json")); m=d["memory"]
    print(f"  {c:<20} VRAM used {m['peak_device_vram_used_mb']/1024:6.2f} GiB  "
          f"min free {m['vram_min_free_mb']/1024:6.2f} GiB  "
          f"rank VmHWM {m['peak_rank_vmhwm_kb_max']/1048576:6.2f} GiB")
except Exception: pass
EOF
  done
done
