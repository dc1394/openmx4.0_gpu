#!/bin/bash
# Aggregate the 24 sidia444_cluster repetitions (4 node counts x GPU/CPU x 3,
# 48 ranks/node) run with the host-memory-reduced binary (dd40d46 + 28ae7f8).
#
# Reads the one-line "TIMING <case> total=.. dft=.. diag=.." record each job
# writes into its own joblog and reports, per (node count, path), the mean /
# sample SD of Total and Diagonalization over the three repetitions, plus the
# GPU speedup per node count.  Utot agreement and host peaks follow.
#
# Run from work/:  ./summarize_s444cl_reps.sh
set -u
cd "$(dirname "$0")" || exit 1

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

# stdin: rows "case total dft diag sh status"; $1 = label, $2 = space-free tag
# (gpu/cpu), $3 = node count.  Emits "MEAN tag nnode phase mean sd" on stderr.
stats() {
  awk -v label="$1" -v tag="$2" -v nn="$3" '
    $6=="OK" { n++; for (i=2;i<=5;i++){ s[i]+=$i; q[i]+=$i*$i; if(n==1||$i<lo[i])lo[i]=$i; if(n==1||$i>hi[i])hi[i]=$i } }
    $6!="OK" { bad = bad " " $1 }
    END {
      if (n==0) { printf "  %-22s no usable runs:%s\n", label, bad; exit }
      printf "  %-22s (n=%d)%s\n", label, n, (bad=="" ? "" : "   [excluded:" bad " ]")
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

echo "sidia444_cluster node scaling, 48 ranks/node, 3 repetitions per point"
echo "host-memory-reduced binary; Max_Time column (slowest rank), seconds"
echo

echo "--- per-run ---"
printf "%-16s %10s %10s %10s %10s  %s\n" case total DFT Diag Set_Ham status
for n in 1 2 3 4; do
  collect "s444cl_${n}n_g1 s444cl_${n}n_g2 s444cl_${n}n_g3 s444cl_${n}n_c1 s444cl_${n}n_c2 s444cl_${n}n_c3" | \
    awk '{printf "%-16s %10s %10s %10s %10s  %s\n", $1,$2,$3,$4,$5,$6}'
done
echo

echo "--- statistics (mean +- sample SD over 3 runs) ---"
: > /tmp/.s444_means
for n in 1 2 3 4; do
  collect "s444cl_${n}n_g1 s444cl_${n}n_g2 s444cl_${n}n_g3" | \
    stats "${n} node(s), GPU+MPS" gpu "$n" 2>>/tmp/.s444_means
  collect "s444cl_${n}n_c1 s444cl_${n}n_c2 s444cl_${n}n_c3" | \
    stats "${n} node(s), CPU elpa2" cpu "$n" 2>>/tmp/.s444_means
done
echo

# Ratio of two measured means; its uncertainty is the two relative SDs added in
# quadrature -- quoting the ratio bare would imply a precision neither run has.
echo "--- GPU speedup per node count (CPU mean / GPU mean, +- in quadrature) ---"
awk '
  $1=="MEAN" && $2=="gpu" { gm[$3" "$4]=$5; gs[$3" "$4]=$6 }
  $1=="MEAN" && $2=="cpu" { cm[$3" "$4]=$5; cs[$3" "$4]=$6 }
  END {
    for (n=1;n<=4;n++) {
      for (p=1;p<=2;p++) {
        ph = (p==1) ? "Total" : "Diagonalization"
        k = n" "ph
        if ((k in gm) && (k in cm) && gm[k]>0 && cm[k]>0) {
          r = cm[k]/gm[k]
          rel = sqrt((gs[k]/gm[k])^2 + (cs[k]/cm[k])^2)
          printf "  %d node(s)  %-16s %6.2fx +- %.2f\n", n, ph, r, r*rel
        }
      }
    }
  }' /tmp/.s444_means
rm -f /tmp/.s444_means
echo

echo "--- Utot (within a (path, node count): identical; across: ~12 digits) ---"
for n in 1 2 3 4; do
  for c in s444cl_${n}n_g1 s444cl_${n}n_g2 s444cl_${n}n_g3 s444cl_${n}n_c1 s444cl_${n}n_c2 s444cl_${n}n_c3; do
    u=$(grep -E "Utot\." "${c}/${c}.out" 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "  %-16s %s\n" "$c" "${u:-(no .out)}"
  done
done
echo

echo "--- peak host memory, node 0 (GiB); old refs: GPU 1n OOM at >=99, CPU 1n 110 ---"
for n in 1 2 3 4; do
  for r in 1 2 3; do
    c=s444cl_${n}n_g${r}
    printf "  %-16s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.smi 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
  done
  for r in 1 2 3; do
    c=s444cl_${n}n_c${r}
    printf "  %-16s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.mem 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
  done
done
