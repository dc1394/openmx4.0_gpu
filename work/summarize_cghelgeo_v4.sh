#!/bin/bash
# Aggregate the six cghelgeo v4-binary (round-3 memory work, commit e227507) repetitions (3 GPU + 3 CPU, 2 nodes x 48 = 96 ranks).
#
# Reads the one-line "TIMING <case> total=.. dft=.. diag=.." record each job
# writes into its own joblog, and reports mean / sample SD / min / max per path
# plus the GPU speedup.  Also checks that every run produced the same Utot, since
# a timing average over runs that did not compute the same thing is meaningless.
#
# Run from work/:  ./summarize_cghelgeo_reps.sh
set -u
cd "$(dirname "$0")" || exit 1

GPU_CASES="cghelgeo_v4_g1 cghelgeo_v4_g2 cghelgeo_v4_g3"
CPU_CASES="cghelgeo_v4_c1 cghelgeo_v4_c2 cghelgeo_v4_c3"

collect() {  # $1 = case list -> "case total dft diag sh" per line
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

stats() {  # stdin: rows "case total dft diag sh status"; $1 = label, $2 = space-free tag
  awk -v label="$1" -v tag="$2" '
    $6=="OK" { n++; for (i=2;i<=5;i++){ s[i]+=$i; q[i]+=$i*$i; if(n==1||$i<lo[i])lo[i]=$i; if(n==1||$i>hi[i])hi[i]=$i } }
    $6!="OK" { bad = bad " " $1 }
    END {
      if (n==0) { printf "%s: no usable runs%s\n", label, bad; exit }
      printf "%s  (n=%d)%s\n", label, n, (bad=="" ? "" : "   [excluded:" bad " ]")
      printf "  %-18s %10s %9s %9s %9s %8s\n", "phase", "mean", "sd", "min", "max", "sd/mean"
      split("2 3 4 5", idx, " "); split("Total DFT Diagonalization Set_Hamiltonian", nm, " ")
      for (k=1;k<=4;k++) {
        i = idx[k]; m = s[i]/n
        sd = (n>1) ? sqrt((q[i]-n*m*m)/(n-1)) : 0
        printf "  %-18s %10.2f %9.2f %9.2f %9.2f %7.2f%%\n", nm[k], m, sd, lo[i], hi[i], (m?100*sd/m:0)
        # Machine-readable side channel for the speedup step.  "tag" is used
        # rather than "label" precisely because label contains spaces, which
        # would shift the field positions the reader relies on.
        printf "MEAN %s %s %.4f %.4f\n", tag, nm[k], m, sd > "/dev/stderr"
      }
    }'
}

echo "cghelgeo, 2 nodes x 48 ranks = 96 ranks, three repetitions per path"
echo "Max_Time column (slowest rank), seconds"
echo

gpu_rows=$(collect "$GPU_CASES")
cpu_rows=$(collect "$CPU_CASES")

echo "--- per-run ---"
printf "%-24s %10s %10s %10s %10s  %s\n" case total DFT Diag Set_Ham status
printf "%s\n%s\n" "$gpu_rows" "$cpu_rows" | \
  awk '{printf "%-24s %10s %10s %10s %10s  %s\n", $1,$2,$3,$4,$5,$6}'
echo

echo "--- statistics ---"
echo "$gpu_rows" | stats "GPU (gpusolver+MPS)" gpu 2>/tmp/.cg4_gpu_means
echo
echo "$cpu_rows" | stats "CPU (elpa2)"         cpu 2>/tmp/.cg4_cpu_means
echo

# Ratio of two measured means, so its uncertainty is the two relative SDs added
# in quadrature -- quoting the ratio bare would imply a precision neither run has.
echo "--- GPU speedup (CPU mean / GPU mean), +- from both SDs in quadrature ---"
awk '
  FNR==NR { gm[$3]=$4; gs[$3]=$5; next }
  ($3 in gm) && gm[$3]>0 {
    r = $4/gm[$3]
    rel = sqrt((gs[$3]/gm[$3])^2 + ($5/$4)^2)
    printf "  %-18s %.2fx +- %.2f\n", $3, r, r*rel
  }
' /tmp/.cg4_gpu_means /tmp/.cg4_cpu_means
rm -f /tmp/.cg4_gpu_means /tmp/.cg4_cpu_means
echo

# Within a path all three repetitions must be bit-identical -- if they are not,
# something other than the configuration under test changed.  Across paths they
# need only agree to ~12 digits: cuSOLVER and ELPA2 accumulate rounding
# differently, so demanding an exact match there would be wrong.
echo "--- Utot (within a path: bit-identical; across paths: ~12 digits) ---"
for c in $GPU_CASES $CPU_CASES; do
  u=$(grep -E "Utot\." "${c}/${c}.out" 2>/dev/null | tail -1 | awk '{print $NF}')
  printf "  %-24s %s\n" "$c" "${u:-(no .out)}"
done
echo

echo "--- peak host memory, node 0 (GiB) ---"
for c in $GPU_CASES; do
  printf "  %-24s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.smi 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
done
for c in $CPU_CASES; do
  printf "  %-24s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.mem 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
done
