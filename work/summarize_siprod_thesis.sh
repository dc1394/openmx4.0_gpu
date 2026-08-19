#!/bin/bash
# Aggregate the THESIS production matrix measured from the v2.0_thesis tag
# (plan v2.6 sec. 5/8.1/8.3; re-measurement of steps 12/14/15 mandated by
# sec. 3: production tables come from the frozen tag build).
#   Cases: sip_<combo>_<cfg><rep>, thesis rep band:
#     c48: 11-13   o: 11-15   g: 11-15   om (bnc216/cnc216): 11-13
# Layer 1: S_GPU = T(c48)/T(o).  Layer 3: S_G8 = T(o)/T(g).  MPS: T(om)/T(o).
# Ratio errors: relative SDs in quadrature; error bar crossing 1 -> "n.s.".
# The manifest of every run is checked against the path-quality conditions
# of plan sec. 10.2 (right column of the table below must be all OK).
# Run from work/:  ./summarize_siprod_thesis.sh
set -u
cd "$(dirname "$0")" || exit 1

COMBOS="bcol216 bcol300 bnc216 ccol216 ccol360 cnc216 cnc300"

reps() {  # $1=combo $2=cfg  (thesis band; slow-node discards get new reps)
  case "$2" in
    c48|om) echo "11 12 13";;
    o|g)    echo "11 12 13 14 15";;
  esac
}
has_om() { case "$1" in bnc216|cnc216) return 0;; *) return 1;; esac; }

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
if cfg in ("o","g","om"):
    if ds["path"]!="gpusolver-gpu-dense": bad.append("path="+ds["path"])
    if ds["cpu_solves"]!=0: bad.append("cpu_solves=%d"%ds["cpu_solves"])
if cfg=="g":
    fb=g8["d_fallbacks"]+g8["z_fallbacks"]
    if fb: bad.append("g8_fb=%d"%fb)
    if not g8["enabled"]: bad.append("g8_off")
    if g8["d_calls"]+g8["z_calls"]==0: bad.append("g8_nocalls")
if cfg in ("o","om") and (g8["d_calls"]+g8["z_calls"])!=0: bad.append("g8_ran")
if cfg=="c48" and d["input"]["eigen_lib"]!="elpa2": bad.append("lib="+d["input"]["eigen_lib"])
mps=d["layout"]["mps"]["detected"]
if cfg in ("o","g") and not mps: bad.append("no-mps")
# om: the manifest's detected flag keys on the exported pipe-dir env var,
# which the MPS-off jobs still carry without ever starting a daemon; the
# authoritative evidence is the job's own control_pipe assertion, checked
# by the shell caller below (MPSOFF-EVIDENCE).
print("OK" if not bad else ";".join(bad))
EOF
}

echo "THESIS production matrix (v2.0_thesis tag build): Layer 1 (CPU vs GPU),"
echo "Layer 3 (cuBLAS vs GEMMul8), MPS ablation.  48 ranks (-nt 1), 1 node;"
echo "CPU = elpa2 flat MPI (best config).  Fixed 25 SCF (criterion 1e-15),"
echo "Max_Time column (s).  Path quality: every run's manifest must satisfy"
echo "plan sec. 10.2 (right column all OK)."
echo
echo "--- per-run (with manifest verdict) ---"
printf "%-20s %10s %10s %10s %10s  %-6s %s\n" case total DFT Diag Set_Ham status manifest
for combo in $COMBOS; do
  for cfg in c48 o g om; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    for r in $(reps "$combo" "$cfg"); do
      c="sip_${combo}_${cfg}${r}"
      collect "$c" | awk '{printf "%-20s %10s %10s %10s %10s  %-6s", $1,$2,$3,$4,$5,$6}'
      v=$(manifest_check "$c" "$cfg")
      if [ "$cfg" = om ]; then
        # MPSOFF-EVIDENCE: the om job aborts unless every node shows
        # control_pipe=no; re-verify that assertion ran
        if grep -q "control_pipe=no" "$c/$c.env" 2>/dev/null; then v="$v mps-off-verified"
        else v="$v MPSOFF-UNPROVEN"; fi
      fi
      printf " %s\n" "$v"
    done
  done
done
echo

echo "--- statistics (mean +- sample SD; CV flag at >2%) ---"
: > /tmp/.sipt_means
for combo in $COMBOS; do
  echo "  [$combo]"
  for cfg in c48 o g om; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    cl=""; for r in $(reps "$combo" "$cfg"); do cl="$cl sip_${combo}_${cfg}${r}"; done
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
      }' 2>>/tmp/.sipt_means
  done
done
echo

echo "--- Layer 1 / Layer 3 / MPS ratios (Total and Diag; +- in quadrature) ---"
awk '
  $1=="MEAN" { m[$2" "$3" "$4]=$5; sd[$2" "$3" "$4]=$6 }
  function ratio(combo, a, b, label, ph,   ka, kb, r, rel, sig) {
    ka=combo" "a" "ph; kb=combo" "b" "ph
    if (!((ka in m)&&(kb in m)) || m[kb]<=0) return
    r=m[ka]/m[kb]; rel=sqrt((sd[ka]/m[ka])^2+(sd[kb]/m[kb])^2)
    sig=(r-1>r*rel || 1-r>r*rel) ? "" : "  [not significant]"
    printf "    %-26s %-6s %6.3f +- %.3f%s\n", label, ph, r, r*rel, sig
  }
  END {
    split("bcol216 bcol300 bnc216 ccol216 ccol360 cnc216 cnc300", C," ")
    for (i=1;i<=7;i++) {
      printf "  [%s]\n", C[i]
      ratio(C[i], "c48", "o", "Layer1 CPU/GPU speedup", "Total")
      ratio(C[i], "c48", "o", "Layer1 CPU/GPU speedup", "Diag")
      ratio(C[i], "g", "o", "Layer3 G8/cuBLAS (>1 slower)", "Total")
      ratio(C[i], "g", "o", "Layer3 G8/cuBLAS (>1 slower)", "Diag")
      ratio(C[i], "om", "o", "MPSoff/MPSon (>1 = MPS helps)", "Total")
      ratio(C[i], "om", "o", "MPSoff/MPSon (>1 = MPS helps)", "Diag")
    }
  }' /tmp/.sipt_means
rm -f /tmp/.sipt_means
echo

echo "--- Utot consistency (per combo: each cfg's last-rep value) ---"
for combo in $COMBOS; do
  printf "  %-8s" "$combo"
  for cfg in c48 o g om; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    r=$(reps "$combo" "$cfg" | awk '{print $NF}')
    u=$(grep -E "Utot\." "sip_${combo}_${cfg}${r}/sip_${combo}_${cfg}${r}.out" 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "  %s=%s" "$cfg" "${u:-?}"
  done
  printf "\n"
done
echo

echo "--- node landings (slow: bnode013, bnode033) ---"
for combo in $COMBOS; do
  for cfg in c48 o g om; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    for r in $(reps "$combo" "$cfg"); do
      c="sip_${combo}_${cfg}${r}"
      nl=$(awk '/^=== nodes ===/{f=1;next} /^=== /{f=0} f&&NF==1{printf "%s ",$1}' "$c/$c.env" 2>/dev/null)
      flag=""; case "$nl" in *bnode013*|*bnode033*) flag="  <-- SLOW NODE: discard, re-measure";; esac
      printf "  %-20s %s%s\n" "$c" "${nl:-(no .env)}" "$flag"
    done
  done
done
echo

echo "--- peak memory (manifest: device VRAM / rank VmHWM; .smi: node RSS) ---"
for combo in $COMBOS; do
  for cfg in o g om; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    r=$(reps "$combo" "$cfg" | awk '{print $1}')
    c="sip_${combo}_${cfg}${r}"
    python3 - "$c" <<'EOF' 2>/dev/null
import json,sys
c=sys.argv[1]
try:
    d=json.load(open(f"{c}/{c}.manifest.json")); m=d["memory"]
    print(f"  {c:<20} VRAM {m['peak_device_vram_used_mb']/1024:6.1f} GiB  "
          f"rank VmHWM {m['peak_rank_vmhwm_kb_max']/1048576:6.2f} GiB")
except Exception: pass
EOF
    rss=$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' "$c/$c.smi" 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)
    [ -n "$rss" ] && printf "  %-20s node RSS %s GiB (.smi)\n" "$c" "$rss"
  done
done
