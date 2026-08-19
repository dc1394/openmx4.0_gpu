#!/bin/bash
# Aggregate the dia64_dc-lno node-scaling benchmark:
# {1,2,3,4} nodes x {g: gpusolver + gemmul8 on, o: gpusolver + gemmul8 off,
# c: CPU elpa2} x 3 reps, all -npernode 48.
# Binary: gpusolver2 merge (commit f5fb5b3).
#
# System: 64 C atoms, C6.0-s2p2d1 -> 832 basis functions, scf.EigenvalueSolver
# dc-lno, Gamma-only, 50 SCF max to 1e-10, orderN.HoppingRanges 6.0 Ang,
# MD.Type Opt with MD.maxIter 1 (one geometry step, so forces are computed).
#
# Supersedes the dia64dc_1n48_* runs, which used the same deck with
# MD.Type nomd and are excluded here.
#
# Two things to read before the ratios:
#
#  1. DC-LNO is not the cluster solver.  It diagonalizes one dense local matrix
#     per atom cluster, and Divide_Conquer_LNO.c sends a cluster to the GPU
#     only when its local dimension is >= 800 (OPENMX_DCLNO_GPU_THRESHOLD).
#     The GPU engagement section below is the evidence that this happened; if
#     it had not, all three configs would have run the same CPU eigensolver.
#
#  2. Set_Allocate_Atom2CPU.c caps the atom-owning ranks at
#     min(numprocs, atomnum) = min(48n, 64).  So 1 node has 48 owners and
#     2, 3 and 4 nodes all have 64.  Gains past 2 nodes come from spreading
#     those 64 owners thinner (more bandwidth and cache per owner, more GPUs),
#     not from more owners.  The strong-scaling section is still a real
#     measurement of what the user gets per node; it is just not the usual
#     "more ranks on the work" story.
#
# Run from work/:  ./summarize_dia64dc_nodes.sh
set -u
cd "$(dirname "$0")" || exit 1

reps() {  # $1 = node count, $2 = cfg -> rep numbers to use
  # 1n o1 drew slow node bnode013 and was re-measured as o4.
  if [ "$1" = 1 ] && [ "$2" = o ]; then echo "2 3 4"
  else echo "1 2 3"; fi
}

cases() { for r in $(reps "$1" "$2"); do printf "dia64dc_%sn_%s%s " "$1" "$2" "$r"; done; }

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

echo "dia64_dc-lno node scaling, 48 ranks/node, 3 repetitions per point"
echo "64 C atoms, C6.0-s2p2d1 -> 832 basis functions, scf.EigenvalueSolver dc-lno"
echo "Gamma-only, 50 SCF max to 1e-10, orderN.HoppingRanges 6.0 Ang"
echo "MD.Type Opt, MD.maxIter 1 -- one geometry step, so forces are computed"
echo "binary: gpusolver2 merge, commit f5fb5b3; Max_Time column (slowest rank), s"
echo "configs: g = GPU gpusolver + MPS, scf.gemmul8.enable on (default)"
echo "         o = GPU gpusolver + MPS, scf.gemmul8.enable off (plain cuBLAS FP64)"
echo "         c = CPU elpa2, no GPU reserved"
echo "atom-owning ranks = min(48n, 64): 48 at 1 node, 64 at 2, 3 and 4 nodes."
echo "1-node g8-off uses reps o2/o3/o4; o1 drew slow node bnode013 and was"
echo "discarded and re-measured as o4, per standing convention."
echo "Supersedes dia64dc_1n48_*, which ran the same deck with MD.Type nomd."
echo

echo "--- per-run ---"
printf "%-18s %10s %10s %10s %10s  %s\n" case total DFT Diag Set_Ham status
for n in 1 2 3 4; do
  collect "$(cases $n g) $(cases $n o) $(cases $n c)" | \
    awk '{printf "%-18s %10s %10s %10s %10s  %s\n", $1,$2,$3,$4,$5,$6}'
done
echo

echo "--- statistics (mean +- sample SD over 3 runs) ---"
: > /tmp/.dia64n_means
for n in 1 2 3 4; do
  collect "$(cases $n g)" | stats "${n} node(s), GPU g8-on"  g "$n" 2>>/tmp/.dia64n_means
  collect "$(cases $n o)" | stats "${n} node(s), GPU g8-off" o "$n" 2>>/tmp/.dia64n_means
  collect "$(cases $n c)" | stats "${n} node(s), CPU elpa2"  c "$n" 2>>/tmp/.dia64n_means
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
    for (i=1;i<=4;i++) for (p=1;p<=2;p++) {
      ph = (p==1) ? "Total" : "Diagonalization"
      ratio("c", "g", "CPU / GPU(g8-on)  [speedup]", i, ph)
      ratio("c", "o", "CPU / GPU(g8-off) [speedup]", i, ph)
      ratio("g", "o", "GPU g8-on / g8-off", i, ph)
    }
  }' /tmp/.dia64n_means
echo
echo "  Note on the last ratio: >1 means GEMMul8 on is SLOWER than plain cuBLAS."
echo "  GEMMul8 intercepts only the three DGEMMs per cluster solve in"
echo "  DCLNO_Solve_Col_GpuSolver; the cusolverDn eigendecomposition between"
echo "  them is untouched, so its leverage on this workload is limited."
echo

echo "--- strong scaling (T(1 node) / T(n nodes)) ---"
awk '
  $1=="MEAN" { m[$2" "$3" "$4]=$5 }
  END {
    split("g o c", tg, " "); split("GPU-g8-on GPU-g8-off CPU-elpa2", nmt, " ")
    for (p=1;p<=2;p++) {
      ph = (p==1) ? "Total" : "Diagonalization"
      for (t=1;t<=3;t++) {
        one = tg[t]" 1 "ph
        if (!(one in m)) continue
        printf "  %-14s %-16s", nmt[t], ph
        for (i=1;i<=4;i++) {
          k = tg[t]" "i" "ph
          if (k in m) printf "  %dn %5.2fx", i, m[one]/m[k]; else printf "  %dn     -", i
        }
        printf "\n"
      }
    }
    printf "  (ideal 1/2/3/4; owners are capped at 64 from 2 nodes on, so the\n"
    printf "   gain past 2 nodes is thinner packing per node, not more owners)\n"
  }' /tmp/.dia64n_means
rm -f /tmp/.dia64n_means
echo

# The decisive check.  If GPU utilisation never rises, DC-LNO never cleared its
# dimension >= 800 gate and g/o ran the same CPU eigensolver as elpa2.
echo "--- GPU engagement evidence (g/o only) ---"
for cfg in g o; do
  for n in 1 2 3 4; do
    for c in $(cases $n $cfg); do
      peak=$(grep -ohE '^[0-9]+ %, [0-9]+ MiB' "${c}/${c}.smi" 2>/dev/null | sort -t, -k2 -rn | head -1)
      util=$(grep -ohE '^[0-9]+ %' "${c}/${c}.smi" 2>/dev/null | tr -d ' %' | sort -rn | head -1)
      acc=$(grep -c "GPU-accelerated" "${c}/${c}.joblog" 2>/dev/null)
      printf "  %-18s peak %-16s max util %3s%%   GPU-accel banners %s\n" \
             "$c" "${peak:-(no .smi)}" "${util:--}" "${acc:-0}"
    done
  done
done
echo

echo "--- node(s) each run landed on (slow: bnode013, bnode033) ---"
for n in 1 2 3 4; do
  for cfg in g o c; do
    for c in $(cases $n $cfg); do
      nl=$(awk '/^=== nodes ===/{f=1;next} /^=== /{f=0} f&&NF==1{printf "%s ",$1}' "${c}/${c}.env" 2>/dev/null)
      flag=""; case "$nl" in *bnode013*|*bnode033*) flag="  <-- SLOW NODE";; esac
      printf "  %-18s %s%s\n" "$c" "${nl:-(no .env)}" "$flag"
    done
  done
done
echo

echo "--- Utot ---"
for n in 1 2 3 4; do
  for cfg in g o c; do
    for c in $(cases $n $cfg); do
      u=$(grep -E "Utot\." "${c}/${c}.out" 2>/dev/null | tail -1 | awk '{print $NF}')
      printf "  %-18s %s\n" "$c" "${u:-(no .out)}"
    done
  done
done
echo

# |Maximum force|: MD.Type Opt now prints it in the MD table at 8 dp, and it is
# also recomputed at 12 dp from the coordinates.forces block.
#
# Read this section knowing what the geometry is: ideal diamond, every atom on
# a symmetry site, so the exact forces are zero.  The printed column is
# 0.00000000 for every path and the recomputed one is roundoff at ~1e-11.  That
# confirms no path is grossly wrong but discriminates far less than it would
# for a displaced structure.  Utot above is the stronger accuracy check.
echo "--- Maximum force (Hartree/Bohr): printed (8 dp) / recomputed (12 dp) ---"
: > /tmp/.dia64n_force
for n in 1 2 3 4; do
  for cfg in g o c; do
    for c in $(cases $n $cfg); do
      p=$(grep -a -A4 "MD_iter   SD_scaling" "${c}/${c}.out" 2>/dev/null | awk '$1=="1"&&NF>=5{print $3; exit}')
      r=$(awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{s=$6*$6+$7*$7+$8*$8; if(s>m)m=s} END{if(m>0)printf "%.12f", sqrt(m)}' "${c}/${c}.out" 2>/dev/null)
      printf "  %-18s %-12s %s\n" "$c" "${p:-(none)}" "${r:--}"
      echo "$cfg $n ${r:--}" >> /tmp/.dia64n_force
    done
  done
done
echo
echo "  comparison per node count (recomputed |F|max):"
awk '
  { k=$1" "$2; if (!(k in v)) v[k]=$3; else if (v[k]!=$3) v[k]="REPS-DIFFER" }
  END {
    for (n=1;n<=4;n++) {
      g=v["g "n]; o=v["o "n]; c=v["c "n]
      printf "  %d node(s):  on %s | off %s | CPU %s", n, g, o, c
      if (g==o && o==c && g!="REPS-DIFFER") printf "  -> identical to 1e-12\n"
      else printf "  -> off-on = %+.1e, CPU-on = %+.1e\n", o-g, c-g
    }
  }' /tmp/.dia64n_force
rm -f /tmp/.dia64n_force
echo

echo "--- peak host memory, node 0 (GiB) ---"
for n in 1 2 3 4; do
  for cfg in g o; do
    for c in $(cases $n $cfg); do
      printf "  %-18s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.smi 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
    done
  done
  for c in $(cases $n c); do
    printf "  %-18s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.mem 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
  done
done
