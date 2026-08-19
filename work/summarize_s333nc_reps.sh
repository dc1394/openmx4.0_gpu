#!/bin/bash
# Aggregate the 36 sidia333_nc runs: {1,2,3,4} nodes x
# {g: GPU gemmul8-on, o: GPU gemmul8-off, c: CPU elpa2} x 3 reps, -npernode 48.
# Round-3 binary (commit e227507).
#
# The 1-node g8-on point uses reps g4/g5/g6 (re-measured 2026-08-14 on clean
# nodes); the original triple's g1 ran on slow node bnode013 and was discarded.
#
# Reads the one-line "TIMING <case> total=.. dft=.. diag=.. set_hamiltonian=.."
# record each job writes into its own joblog and reports, per (node count,
# config), the mean / sample SD of Total and Diagonalization over the three
# repetitions, plus per-node-count ratios.  Utot / Maximum-force agreement and
# host peaks follow.
#
# Run from work/:  ./summarize_s333nc_reps.sh
set -u
cd "$(dirname "$0")" || exit 1

reps() {  # $1 = node count, $2 = cfg -> rep numbers to use
  if [ "$1" = 1 ] && [ "$2" = g ]; then echo "4 5 6"; else echo "1 2 3"; fi
}

cases() {  # $1 = node count, $2 = cfg -> space-separated case names
  for r in $(reps "$1" "$2"); do printf "s333nc_%sn_%s%s " "$1" "$2" "$r"; done
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

# stdin: rows "case total dft diag sh status"; $1 = label, $2 = tag (g/o/c),
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

echo "sidia333_nc node scaling, 48 ranks/node, 3 repetitions per point"
echo "216 Si non-collinear (matrix dim 5616), BAND solver k-grid 2x2x2, 25 SCF"
echo "round-3 binary (e227507); Max_Time column (slowest rank), seconds"
echo "configs: g = GPU gpusolver+MPS, gemmul8 on (default) | o = same but"
echo "         scf.gemmul8.enable off (plain cuBLAS) | c = CPU elpa2, no GPU"
echo "1-node g8-on = re-measurement of 2026-08-14 (g4/g5/g6, all on clean nodes:"
echo "bnode002/025/026); the original g1/g2/g3 triple is in the .bak file -- its"
echo "g1 (283.1 s) ran on slow node bnode013 and was discarded per instruction."
echo

echo "--- per-run ---"
printf "%-16s %10s %10s %10s %10s  %s\n" case total DFT Diag Set_Ham status
for n in 1 2 3 4; do
  collect "$(cases $n g) $(cases $n o) $(cases $n c)" | \
    awk '{printf "%-16s %10s %10s %10s %10s  %s\n", $1,$2,$3,$4,$5,$6}'
done
echo

echo "--- statistics (mean +- sample SD over 3 runs) ---"
: > /tmp/.s333nc_means
for n in 1 2 3 4; do
  collect "$(cases $n g)" | stats "${n} node(s), GPU g8-on" g "$n" 2>>/tmp/.s333nc_means
  collect "$(cases $n o)" | stats "${n} node(s), GPU g8-off" o "$n" 2>>/tmp/.s333nc_means
  collect "$(cases $n c)" | stats "${n} node(s), CPU elpa2" c "$n" 2>>/tmp/.s333nc_means
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
        ratio("c", "g", "CPU / GPU(g8-on)  [speedup]", i, ph)
        ratio("c", "o", "CPU / GPU(g8-off) [speedup]", i, ph)
        ratio("o", "g", "GPU g8-off / g8-on", i, ph)
      }
    }
  }' /tmp/.s333nc_means
rm -f /tmp/.s333nc_means
echo

echo "--- Utot (old-binary refs: GPU -887.422830856493, CPU -887.422830856506) ---"
for n in 1 2 3 4; do
  for cfg in g o c; do
    for c in $(cases $n $cfg); do
      u=$(grep -E "Utot\." "${c}/${c}.out" 2>/dev/null | tail -1 | awk '{print $NF}')
      printf "  %-16s %s\n" "$c" "${u:-(no .out)}"
    done
  done
done
echo

# |Maximum force| = max over atoms of sqrt(fx^2+fy^2+fz^2), non-fixed
# components only (MD_pac.c).  The MD table in .out prints it at 8 dp, so it
# is also recomputed from the 12-dp forces in the coordinates.forces block to
# expose any gemmul8-on / gemmul8-off / CPU difference hiding below 8 dp.
echo "--- Maximum force (Hartree/Bohr): gemmul8-on vs gemmul8-off vs CPU ---"
echo "  per run: printed |Maximum force| (8 dp) / recomputed from 12-dp forces"
: > /tmp/.s333nc_force
for n in 1 2 3 4; do
  for cfg in g o c; do
    for c in $(cases $n $cfg); do
      p=$(grep -a -A4 "MD_iter   SD_scaling" "${c}/${c}.out" 2>/dev/null | awk '$1=="1"&&NF>=5{print $3; exit}')
      r=$(awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{s=$6*$6+$7*$7+$8*$8; if(s>m)m=s} END{if(m>0)printf "%.12f", sqrt(m)}' "${c}/${c}.out" 2>/dev/null)
      printf "  %-16s %-12s %s\n" "$c" "${p:-(no .out)}" "${r:--}"
      echo "$cfg $n ${r:--}" >> /tmp/.s333nc_force
    done
  done
done
echo
echo "  three-way comparison per node count (recomputed |F|max):"
awk '
  { k=$1" "$2; if (!(k in v)) v[k]=$3; else if (v[k]!=$3) v[k]="REPS-DIFFER" }
  END {
    for (n=1;n<=4;n++) {
      g=v["g "n]; o=v["o "n]; c=v["c "n]
      printf "  %d node(s):  on %s | off %s | CPU %s", n, g, o, c
      if (g==o && o==c && g!="REPS-DIFFER") printf "  -> identical to 1e-12\n"
      else printf "  -> off-on = %+.1e, CPU-on = %+.1e\n", o-g, c-g
    }
  }' /tmp/.s333nc_force
rm -f /tmp/.s333nc_force
echo

echo "--- peak host memory, node 0 (GiB); old-binary GPU 1n48 completed fine ---"
for n in 1 2 3 4; do
  for cfg in g o; do
    for c in $(cases $n $cfg); do
      printf "  %-16s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.smi 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
    done
  done
  for c in $(cases $n c); do
    printf "  %-16s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.mem 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
  done
done
