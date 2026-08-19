#!/bin/bash
# Aggregate the H100 1-node production matrix (plan v2.6 sec. 8.1/8.3,
# steps 12/14/15): sip_<s><mode><atoms>_<cfg><rep>
#   cfgs: c48 = CPU elpa2 flat 48 (3 reps)   o = GPU cuBLAS + MPS (5 reps)
#         g  = GPU GEMMul8 + MPS (5 reps)    om = GPU cuBLAS, MPS OFF (3 reps)
# Combos: bcol216 bcol300 bnc216 ccol216 ccol360 cnc216 cnc300
# Layer 1: S_GPU = T(c48)/T(o).   Layer 3: S_G8 = T(o)/T(g).
# MPS effect (plan 12): T(om)/T(o)  (>1 = MPS helps).
# Ratio errors: relative SDs added in quadrature; a ratio whose error bar
# crosses 1 is reported as "no significant difference".
# Rep lists in reps() so slow-node discards are one edit.
# Run from work/:  ./summarize_siprod.sh
set -u
cd "$(dirname "$0")" || exit 1

COMBOS="bcol216 bcol300 bnc216 ccol216 ccol360 cnc216 cnc300"

reps() {  # $1=combo $2=cfg
  # ccol216 c48 carries 5 reps (Total CV 2.2% at n=3 -> +2 per the CV rule).
  if [ "$1" = ccol216 ] && [ "$2" = c48 ]; then echo "1 2 3 4 5"; return; fi
  case "$2" in
    c48|om) echo "1 2 3";;
    o|g)    echo "1 2 3 4 5";;
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

desc() {  # $1=combo
  case "$1" in
    bcol216) echo "band col 216 atoms (n=2808)";;
    bcol300) echo "band col 300 atoms (n=3900, representative)";;
    bnc216)  echo "band NC 216 atoms (2n=5616, representative)";;
    ccol216) echo "cluster col 216 atoms (n=2808)";;
    ccol360) echo "cluster col 360 atoms (n=4680, representative)";;
    cnc216)  echo "cluster NC 216 atoms (2n=5616)";;
    cnc300)  echo "cluster NC 300 atoms (2n=7800, representative)";;
  esac
}

echo "H100 1-node production matrix: Layer 1 (CPU vs GPU), Layer 3 (cuBLAS"
echo "vs GEMMul8), MPS ablation.  48 ranks (-nt 1) everywhere; CPU = elpa2"
echo "flat MPI (best config, step-10 preflights: 48 beat 36 by 8-23%)."
echo "Fixed 25 SCF (criterion 1e-15), MD.Type Opt x1, Max_Time column (s)."
echo "binary: openmx cd5f0d5 + GEMMul8 v3.2.0, md5 962f8d2519c2e6aa5a6295513f76fee9"
echo
echo "--- per-run ---"
printf "%-20s %10s %10s %10s %10s  %s\n" case total DFT Diag Set_Ham status
for combo in $COMBOS; do
  for cfg in c48 o g om; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    cl=""; for r in $(reps "$combo" "$cfg"); do cl="$cl sip_${combo}_${cfg}${r}"; done
    collect "$cl" | awk '{printf "%-20s %10s %10s %10s %10s  %s\n", $1,$2,$3,$4,$5,$6}'
  done
done
echo

echo "--- statistics (mean +- sample SD; CV flag at >2%) ---"
: > /tmp/.sip_means
for combo in $COMBOS; do
  echo "  [$combo: $(desc "$combo")]"
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
      }' 2>>/tmp/.sip_means
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
  }' /tmp/.sip_means
rm -f /tmp/.sip_means
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

echo "--- Utot spread across reps within each cfg (max-min, Hartree) ---"
for combo in $COMBOS; do
  for cfg in c48 o g om; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    for r in $(reps "$combo" "$cfg"); do
      grep -E "Utot\." "sip_${combo}_${cfg}${r}/sip_${combo}_${cfg}${r}.out" 2>/dev/null | tail -1 | awk -v t="$combo $cfg" '{print t, $NF}'
    done
  done
done | awk '
  { key=$1" "$2; v=$3+0; if (!(key in lo) || v<lo[key]) lo[key]=v; if (!(key in hi) || v>hi[key]) hi[key]=v; n[key]++ }
  END { for (k in n) if (hi[k]-lo[k] > 0) printf "  %-16s spread %.1e over %d reps\n", k, hi[k]-lo[k], n[k];
        print "  (configs not listed: bitwise identical across reps)" }'
echo

echo "--- GPU engagement / fallback (first line per GPU case) ---"
for combo in $COMBOS; do
  for cfg in o g om; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    for r in $(reps "$combo" "$cfg"); do
      c="sip_${combo}_${cfg}${r}"
      case "$combo" in
        b*) ev=$(grep -oE "GPU device [0-9]+: [0-9]+ k-owner rank" "$c/$c.joblog" 2>/dev/null | sort -u | head -1)
            fb=$(grep -ciE "cannot fit" "$c/$c.joblog" 2>/dev/null);;
        c*) ev=$(grep -oE "<Set_Hamiltonian> GPU device [0-9]+: [0-9]+ Hamiltonian rank" "$c/$c.joblog" 2>/dev/null | head -1)
            fb=$(grep -cE "<Cluster_DFT_(Col|NonCol)>" "$c/$c.joblog" 2>/dev/null);;
      esac
      printf "  %-20s [%s] fb=%s\n" "$c" "${ev:-?}" "${fb:-?}"
    done
  done
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

echo "--- peak host RSS / peak GPU mem ---"
for combo in $COMBOS; do
  for cfg in c48 o g om; do
    [ "$cfg" = om ] && ! has_om "$combo" && continue
    for r in $(reps "$combo" "$cfg"); do
      c="sip_${combo}_${cfg}${r}"
      rss=$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' "$c/$c.smi" "$c/$c.mem" 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)
      gm=$(grep -oE '^[0-9]+ %, [0-9]+ MiB' "$c/$c.smi" 2>/dev/null | awk -F'[ ,]+' '{print $3}' | sort -rn | head -1)
      printf "  %-20s RSS %-9s GPU %s MiB\n" "$c" "${rss:-?}" "${gm:--}"
    done
  done
done
