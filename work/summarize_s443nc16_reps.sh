#!/bin/bash
# Aggregate the -npernode 16 re-measurement of sidia443_nc_cluster:
# {1,2,3,4} nodes x {g: gpusolver + gemmul8 on, c: CPU elpa2,
# d: gpusolver2 = distributed ELPA 2026.02 + COSMA} x 3 reps.
#
# Unlike the 36-rank table (work/s443nc_reps_result.txt), every point here was
# produced by ONE binary: the gpusolver2 merge, commit f5fb5b3.  In the 36-rank
# table the g/o/c columns came from e227507 and only the d column from f5fb5b3,
# so a g-vs-d comparison there crossed a binary boundary; here it does not.
#
# The 1-node d point is s443nc_1n16_d2..4, measured 2026-08-15 at exactly this
# rank count on this binary (its d1 drew slow node bnode013 and was discarded
# per convention); it is reused rather than re-run.  Every other point is
# s443nc_<n>n16_<cfg><rep>, submitted 2026-08-16 as job 912850..912882.
#
# Reads the one-line "TIMING <case> total=.. dft=.. diag=.. set_hamiltonian=.."
# record each job writes into its own joblog and reports, per (node count,
# config), the mean / sample SD of Total and Diagonalization over the three
# repetitions, plus per-node-count ratios.  Utot / Maximum-force agreement,
# host peaks, and the node list of every run follow.
#
# Run from work/:  ./summarize_s443nc16_reps.sh
set -u
cd "$(dirname "$0")" || exit 1

reps() {  # $1 = node count, $2 = cfg -> rep numbers to use
  if [ "$1" = 1 ] && [ "$2" = d ]; then echo "2 3 4"
  else echo "1 2 3"; fi
}

cases() {  # $1 = node count, $2 = cfg -> space-separated case names
  for r in $(reps "$1" "$2"); do printf "s443nc_%sn16_%s%s " "$1" "$2" "$r"; done
}

collect() {  # $1 = case list -> "case total dft diag sh status" per line
  for c in $1; do
    line=$(grep -hm1 "^TIMING ${c} " "${c}/${c}.joblog" 2>/dev/null)
    if [ -z "$line" ]; then
      echo "${c} - - - - MISSING"
    elif echo "$line" | grep -q INCOMPLETE; then
      echo "${c} - - - - INCOMPLETE"
    else
      echo "$line" | sed -E 's/^TIMING ([^ ]+) total=([^ ]+) dft=([^ ]+) diag=([^ ]+) set_hamiltonian=([^ ]+)/\1 \2 \3 \4 \5 OK/'
    fi
  done
}

# stdin: rows "case total dft diag sh status"; $1 = label, $2 = tag (g/c/d),
# $3 = node count.  Emits "MEAN tag nnode phase mean sd" on stderr.
stats() {
  awk -v label="$1" -v tag="$2" -v nn="$3" '
    $6=="OK" { n++; for (i=2;i<=5;i++){ s[i]+=$i; q[i]+=$i*$i; if(n==1||$i<lo[i])lo[i]=$i; if(n==1||$i>hi[i])hi[i]=$i } }
    $6!="OK" { bad = bad " " $1 }
    END {
      if (n==0) { printf "  %-24s no usable runs:%s\n", label, bad; exit }
      printf "  %-24s (n=%d)%s\n", label, n, (bad=="" ? "" : "   [excluded:" bad " ]")
      printf "    %-16s %10s %8s %9s %9s\n", "phase", "mean", "sd", "min", "max"
      split("2 4", idx, " "); split("Total Diagonalization", nm, " ")
      for (k=1;k<=2;k++) {
        i = idx[k]; m = s[i]/n
        sd = (n>1) ? sqrt((q[i]-n*m*m)/(n-1)) : 0
        printf "    %-16s %10.2f %8.2f %9.2f %9.2f\n", nm[k], m, sd, lo[i], hi[i]
        printf "MEAN %s %s %s %.4f %.4f\n", tag, nn, nm[k], m, sd > "/dev/stderr"
      }
    }'
}

echo "sidia443_nc_cluster node scaling, 16 ranks/node, 3 repetitions per point"
echo "384 Si non-collinear (matrix dim 9984), Cluster solver, 25 SCF"
echo "single binary throughout: gpusolver2 merge, commit f5fb5b3"
echo "Max_Time column (slowest rank), seconds"
echo "configs: g = GPU gpusolver + MPS, gemmul8 on (default)"
echo "         c = CPU elpa2, no GPU reserved"
echo "         d = GPU gpusolver2 (distributed ELPA 2026.02 + COSMA), MPS"
echo "Whole nodes are reserved (--cpunum-lhost=48) even though only 16 ranks"
echo "are launched, so the 32 idle cores are not handed to another job."
echo "The 1-node d point reuses s443nc_1n16_d2..4 (2026-08-15, same rank count"
echo "and same binary).  All other points are the 2026-08-16 submission."
echo "Multi-node runs are screened for slow nodes (bnode013/bnode033) on the"
echo "FULL node list in each .env, not just the node the script starts on."
echo

echo "--- per-run ---"
printf "%-18s %10s %10s %10s %10s  %s\n" case total DFT Diag Set_Ham status
for n in 1 2 3 4; do
  collect "$(cases $n g) $(cases $n c) $(cases $n d)" | \
    awk '{printf "%-18s %10s %10s %10s %10s  %s\n", $1,$2,$3,$4,$5,$6}'
done
echo

echo "--- statistics (mean +- sample SD over 3 runs) ---"
: > /tmp/.s443nc16_means
for n in 1 2 3 4; do
  collect "$(cases $n g)" | stats "${n} node(s), GPU g8-on" g "$n" 2>>/tmp/.s443nc16_means
  collect "$(cases $n c)" | stats "${n} node(s), CPU elpa2" c "$n" 2>>/tmp/.s443nc16_means
  collect "$(cases $n d)" | stats "${n} node(s), gpusolver2" d "$n" 2>>/tmp/.s443nc16_means
done
echo

# Ratio of two measured means; its uncertainty is the two relative SDs added in
# quadrature -- quoting the ratio bare would imply a precision neither run has.
echo "--- ratios per node count (mean/mean, +- in quadrature) ---"
awk '
  $1=="MEAN" { m[$2" "$3" "$4]=$5; sd[$2" "$3" "$4]=$6 }
  function ratio(num, den, txt, nn, ph,   kn, kd, r, rel) {
    kn = num" "nn" "ph; kd = den" "nn" "ph
    if ((kn in m) && (kd in m) && m[kn]>0 && m[kd]>0) {
      r = m[kn]/m[kd]
      rel = sqrt((sd[kn]/m[kn])^2 + (sd[kd]/m[kd])^2)
      printf "  %d node(s)  %-28s %-16s %6.2fx +- %.2f\n", nn, txt, ph, r, r*rel
    }
  }
  END {
    for (i=1;i<=4;i++) {
      for (p=1;p<=2;p++) {
        ph = (p==1) ? "Total" : "Diagonalization"
        ratio("c", "g", "CPU / gpusolver   [speedup]", i, ph)
        ratio("c", "d", "CPU / gpusolver2  [speedup]", i, ph)
        ratio("d", "g", "gpusolver2 / gpusolver", i, ph)
      }
    }
  }' /tmp/.s443nc16_means
echo

# Strong scaling relative to this campaign's own 1-node point: the 36-rank
# table cannot answer this for gpusolver2, whose 1-node point there was the
# only 16-rank bar in an otherwise 36-rank set.
echo "--- strong scaling within this campaign (T(1 node) / T(n nodes)) ---"
awk '
  $1=="MEAN" { m[$2" "$3" "$4]=$5; sd[$2" "$3" "$4]=$6 }
  END {
    split("g c d", tg, " "); split("gpusolver CPU-elpa2 gpusolver2", nmt, " ")
    for (p=1;p<=2;p++) {
      ph = (p==1) ? "Total" : "Diagonalization"
      for (t=1;t<=3;t++) {
        one = tg[t]" 1 "ph
        if (!(one in m)) continue
        printf "  %-16s %-16s", nmt[t], ph
        for (i=1;i<=4;i++) {
          k = tg[t]" "i" "ph
          if (k in m) printf "  %dn %5.2fx", i, m[one]/m[k]; else printf "  %dn     -", i
        }
        printf "  (ideal 1/2/3/4)\n"
      }
    }
  }' /tmp/.s443nc16_means
rm -f /tmp/.s443nc16_means
echo

echo "--- nodes each run landed on (slow: bnode013, bnode033) ---"
for n in 1 2 3 4; do
  for cfg in g c d; do
    for c in $(cases $n $cfg); do
      nl=$(awk '/^=== nodes ===/{f=1;next} /^=== /{f=0} f&&NF==1{printf "%s ",$1}' "${c}/${c}.env" 2>/dev/null)
      flag=""
      case "$nl" in *bnode013*|*bnode033*) flag="  <-- SLOW NODE";; esac
      printf "  %-18s %s%s\n" "$c" "${nl:-(no .env)}" "$flag"
    done
  done
done
echo

echo "--- Utot (36-rank campaign reference: -1577.619265912xx) ---"
for n in 1 2 3 4; do
  for cfg in g c d; do
    for c in $(cases $n $cfg); do
      u=$(grep -E "Utot\." "${c}/${c}.out" 2>/dev/null | tail -1 | awk '{print $NF}')
      printf "  %-18s %s\n" "$c" "${u:-(no .out)}"
    done
  done
done
echo

# |Maximum force| = max over atoms of sqrt(fx^2+fy^2+fz^2), non-fixed
# components only (MD_pac.c).  The MD table in .out prints it at 8 dp, so it
# is also recomputed from the 12-dp forces in the coordinates.forces block to
# expose any gpusolver / gpusolver2 / CPU difference hiding below 8 dp.
echo "--- Maximum force (Hartree/Bohr): gpusolver vs gpusolver2 vs CPU ---"
echo "  per run: printed |Maximum force| (8 dp) / recomputed from 12-dp forces"
: > /tmp/.s443nc16_force
for n in 1 2 3 4; do
  for cfg in g c d; do
    for c in $(cases $n $cfg); do
      p=$(grep -a -A4 "MD_iter   SD_scaling" "${c}/${c}.out" 2>/dev/null | awk '$1=="1"&&NF>=5{print $3; exit}')
      r=$(awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{s=$6*$6+$7*$7+$8*$8; if(s>m)m=s} END{if(m>0)printf "%.12f", sqrt(m)}' "${c}/${c}.out" 2>/dev/null)
      printf "  %-18s %-12s %s\n" "$c" "${p:-(no .out)}" "${r:--}"
      echo "$cfg $n ${r:--}" >> /tmp/.s443nc16_force
    done
  done
done
echo
echo "  comparison per node count (recomputed |F|max):"
awk '
  { k=$1" "$2; if (!(k in v)) v[k]=$3; else if (v[k]!=$3) v[k]="REPS-DIFFER" }
  END {
    for (n=1;n<=4;n++) {
      g=v["g "n]; c=v["c "n]; d=v["d "n]
      printf "  %d node(s):  gpusolver %s | CPU %s | gpusolver2 %s", n, g, c, d
      if (g==c && c==d && g!="REPS-DIFFER") printf "  -> identical to 1e-12\n"
      else printf "  -> CPU-gpusolver = %+.1e, g2-gpusolver = %+.1e\n", c-g, d-g
    }
  }' /tmp/.s443nc16_force
rm -f /tmp/.s443nc16_force
echo

echo "--- peak host memory, node 0 (GiB) ---"
for n in 1 2 3 4; do
  for cfg in g d; do
    for c in $(cases $n $cfg); do
      printf "  %-18s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.smi 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
    done
  done
  for c in $(cases $n c); do
    printf "  %-18s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.mem 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
  done
done
