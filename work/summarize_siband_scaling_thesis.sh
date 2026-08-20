#!/bin/bash
# Aggregate the Si diamond BAND limited strong-scaling series
# (research plan v2.6 sec. 7.4 / 8.5 + 8-node thesis extension):
#   sibs_col300_{1,2,4,8}n_{o,g}{reps} and sibs_nc216_{1,2,4,8}n_{o,g}{reps}
#   o = gemmul8 off (cuBLAS FP64), g = gemmul8 on; MPS on; 48 ranks/node.
# Reports mean +- sample SD over the reps, CV (plan: add 2 reps where
# CV > 2%), GEMMul8 ratio per node count, strong scaling S_p = T1/Tp,
# E_p = T1/(p Tp), node-hours p*Tp, k-owner evidence, node landings, Utot.
#
# Rep lists live in reps() so a slow-node discard is one edit, mirroring
# summarize_dia64dc_nodes.sh.
# Run from work/:  ./summarize_siband_scaling_thesis.sh
set -u
cd "$(dirname "$0")" || exit 1

reps() {  # thesis band: 3 reps (11-13) + CV round (14-15, plan 9.2); the
          # 8-node extension then brought EVERY point to n=5 and added the
          # 8n series at n=5; points first flagged CV>2% at n=5 got their
          # one CV round as reps 16-17 (n=7).  nc216_2n_g15 drew bnode013
          # -> discarded, g16 (its CV round was 14+16).
  case "$1:$2:$3" in
    nc216:2:g)   echo "11 12 13 14 16";;
    col300:4:o)  echo "11 12 13 14 15 18 20";;  # 16/17/19 drew bnode033
    nc216:8:g)   echo "11 12 13 14 15 17 18";;  # 16 drew bnode033
    col300:4:g|col300:8:o|nc216:8:o)
                 echo "11 12 13 14 15 16 17";;
    *)           echo "11 12 13 14 15";;
  esac
}
CASEOF() { printf "sibs_%s_%sn_%s%s" "$1" "$2" "$3" "$4"; }

collect() {  # $1 = case list -> "case total dft diag sh status"
  for c in $1; do
    line=$(grep -hm1 "^TIMING ${c} " "${c}/${c}.joblog" 2>/dev/null)
    if [ -z "$line" ]; then echo "${c} - - - - MISSING"
    elif echo "$line" | grep -q INCOMPLETE; then echo "${c} - - - - INCOMPLETE"
    else echo "$line" | sed -E 's/^TIMING ([^ ]+) total=([^ ]+) dft=([^ ]+) diag=([^ ]+) set_hamiltonian=([^ ]+)/\1 \2 \3 \4 \5 OK/'
    fi
  done
}

echo "Si diamond BAND limited strong scaling, 48 ranks/node, -nt 1, MPS on"
echo "col = sibs_col300 (300 atoms, prim 6x5x5, n=3900, spin off)"
echo "nc  = sibs_nc216  (216 atoms, conv 3x3x3, 2n=5616, spin NC)"
echo "8 computed k points; k-owner map 8/4/2/1 per GPU at 1/2/4/8 nodes"
echo "binary: v2.0_thesis tag build, md5 11227640dc6f8a8b194ddcd9ab811917"
echo "deck: fixed 25 SCF (scf.criterion 1e-15), MD.Type Opt x1; Max_Time (s)"
echo "o = scf.gemmul8.enable off (cuBLAS FP64), g = gemmul8 on (INT8 emulation)"
echo
echo "--- per-run ---"
printf "%-22s %10s %10s %10s %10s  %s\n" case total DFT Diag Set_Ham status
for ma in col300 nc216; do
  for N in 1 2 4 8; do for b in o g; do
    cl=""; for r in $(reps "$ma" "$N" "$b"); do cl="$cl $(CASEOF "$ma" "$N" "$b" "$r")"; done
    collect "$cl" | awk '{printf "%-22s %10s %10s %10s %10s  %s\n", $1,$2,$3,$4,$5,$6}'
  done; done
done
echo

echo "--- statistics (mean +- sample SD; CV flag at >2%) ---"
: > /tmp/.sibs_means
for ma in col300 nc216; do
  for N in 1 2 4 8; do for b in o g; do
    cl=""; for r in $(reps "$ma" "$N" "$b"); do cl="$cl $(CASEOF "$ma" "$N" "$b" "$r")"; done
    collect "$cl" | awk -v ma="$ma" -v N="$N" -v b="$b" '
      $6=="OK" { n++; for(i=2;i<=5;i++){s[i]+=$i;q[i]+=$i*$i} }
      $6!="OK" { bad=bad" "$1 }
      END {
        if (n==0) { printf "  %-14s no usable runs:%s\n", ma"_"N"n_"b, bad; exit }
        split("2 4", idx," "); split("Total Diag", nm," ")
        printf "  %-14s (n=%d)%s", ma"_"N"n_"b, n, (bad==""?"":"  [excluded:"bad" ]")
        for (k=1;k<=2;k++) {
          i=idx[k]; m=s[i]/n; sd=(n>1)?sqrt((q[i]-n*m*m)/(n-1)):0
          cv=(m>0)?100*sd/m:0
          printf "   %s %8.2f +-%6.2f (CV %.1f%%%s)", nm[k], m, sd, cv, (cv>2?" **>2%: add 2 reps**":"")
          printf "MEAN %s %s %s %s %.4f %.4f\n", ma, N, b, nm[k], m, sd > "/dev/stderr"
        }
        printf "\n"
      }' 2>>/tmp/.sibs_means
  done; done
done
echo

echo "--- GEMMul8 effect per node count: T(g8-on)/T(g8-off) ---"
awk '
  $1=="MEAN" { m[$2" "$3" "$4" "$5]=$6; sd[$2" "$3" "$4" "$5]=$7 }
  END {
    split("col300 nc216", MA," "); split("1 2 4 8", NN," "); split("Total Diag", PH," ")
    for (a=1;a<=2;a++) for (p=1;p<=2;p++) {
      printf "  %-7s %-5s:", MA[a], PH[p]
      for (n=1;n<=4;n++) {
        kg=MA[a]" "NN[n]" g "PH[p]; ko=MA[a]" "NN[n]" o "PH[p]
        if ((kg in m)&&(ko in m)&&m[ko]>0) {
          r=m[kg]/m[ko]; rel=sqrt((sd[kg]/m[kg])^2+(sd[ko]/m[ko])^2)
          printf "  %dn %.3f+-%.3f", NN[n], r, r*rel
        }
      }
      printf "   (>1 = gemmul8 slower)\n"
    }
  }' /tmp/.sibs_means
echo

echo "--- strong scaling (plan 7.4: S_p=T1/Tp, E_p=T1/(p*Tp), node-hours=p*Tp;"
echo "    Diag rows scale the Diag column alone - the k-parallel part) ---"
awk '
  $1=="MEAN" { m[$2" "$3" "$4" "$5]=$6 }
  END {
    split("col300 nc216", MA," "); split("o g", B," "); split("Total Diag", PH," ")
    for (a=1;a<=2;a++) for (bb=1;bb<=2;bb++) for (ph=1;ph<=2;ph++) {
      t1=m[MA[a]" 1 "B[bb]" "PH[ph]]
      if (t1=="") continue
      printf "  %-7s %s %-5s:", MA[a], (B[bb]=="o"?"cuBLAS ":"GEMMul8"), PH[ph]
      for (p=1;p<=8;p*=2) {
        t=m[MA[a]" "p" "B[bb]" "PH[ph]]
        if (t=="") continue
        if (PH[ph]=="Total") printf "  %dn T=%.1fs S=%.2f E=%.0f%% nh=%.4f", p, t, t1/t, 100*t1/(p*t), p*t/3600
        else                 printf "  %dn T=%.1fs S=%.2f", p, t, t1/t
      }
      printf "\n"
    }
  }' /tmp/.sibs_means
rm -f /tmp/.sibs_means
echo

echo "--- k-owner evidence (first banner per case) + fallback count ---"
for ma in col300 nc216; do
  for N in 1 2 4 8; do for b in o g; do
    for r in $(reps "$ma" "$N" "$b"); do
      c=$(CASEOF "$ma" "$N" "$b" "$r")
      ko=$(grep -oE "GPU device [0-9]+: [0-9]+ k-owner rank" "$c/$c.joblog" 2>/dev/null | sort -u | head -1)
      fb=$(grep -ciE "cannot fit" "$c/$c.joblog" 2>/dev/null || echo 0)
      printf "  %-22s [%s] cfit=%s\n" "$c" "${ko:-?}" "$fb"
    done
  done; done
done
echo

echo "--- node landings (slow: bnode013, bnode033) ---"
for ma in col300 nc216; do
  for N in 1 2 4 8; do for b in o g; do
    for r in $(reps "$ma" "$N" "$b"); do
      c=$(CASEOF "$ma" "$N" "$b" "$r")
      nl=$(awk '/^=== nodes ===/{f=1;next} /^=== /{f=0} f&&NF==1{printf "%s ",$1}' "$c/$c.env" 2>/dev/null)
      flag=""; case "$nl" in *bnode013*|*bnode033*) flag="  <-- SLOW NODE: discard, re-measure new rep";; esac
      printf "  %-22s %s%s\n" "$c" "${nl:-(no .env)}" "$flag"
    done
  done; done
done
echo

echo "--- Utot (per config; must be identical across reps and across node counts) ---"
for ma in col300 nc216; do
  for b in o g; do
    printf "  %s %s:" "$ma" "$b"
    for N in 1 2 4 8; do for r in $(reps "$ma" "$N" "$b"); do
      c=$(CASEOF "$ma" "$N" "$b" "$r")
      printf " %s" "$(grep -E 'Utot\.' "$c/$c.out" 2>/dev/null | tail -1 | awk '{print $NF}')"
    done; done
    printf "\n"
  done
done
echo

echo "--- peak host RSS node0 / peak GPU mem (GiB / MiB) ---"
for ma in col300 nc216; do
  for N in 1 2 4 8; do for b in o g; do
    for r in $(reps "$ma" "$N" "$b"); do
      c=$(CASEOF "$ma" "$N" "$b" "$r")
      rss=$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' "$c/$c.smi" 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)
      gm=$(grep -oE '^[0-9]+ %, [0-9]+ MiB' "$c/$c.smi" 2>/dev/null | awk -F'[ ,]+' '{print $3}' | sort -rn | head -1)
      printf "  %-22s RSS %-9s GPU %s MiB\n" "$c" "${rss:-?}" "${gm:-?}"
    done
  done; done
done
