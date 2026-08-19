#!/bin/bash
# Aggregate the dia64_dc-lno 1-node benchmark:
# 48 ranks x {g: gpusolver + gemmul8 on, o: gpusolver + gemmul8 off,
# c: CPU elpa2} x 3 reps.  Binary: gpusolver2 merge (commit f5fb5b3).
#
# System: 64 C atoms, C6.0-s2p2d1 -> 832 basis functions, dc-lno solver,
# Gamma-only, 50 SCF max to 1e-10, orderN.HoppingRanges 6.0 Ang.
#
# DC-LNO is not the cluster solver: it diagonalizes one dense local matrix per
# atom cluster rather than one global matrix, and Divide_Conquer_LNO.c sends a
# cluster to the GPU only when its local dimension is >= 800
# (OPENMX_DCLNO_GPU_THRESHOLD, default 800).  The "GPU engagement" section
# below is therefore not boilerplate: if the clusters here are smaller than
# that, all three configs ran the same CPU eigensolver and the g/o/c spread is
# noise, not a solver comparison.  Read that section before the ratios.
#
# 64 clusters over 48 ranks does not divide evenly (16 ranks x 2, 32 ranks x 1),
# so this rank count carries an inherent load imbalance.
#
# Run from work/:  ./summarize_dia64dc_reps.sh
set -u
cd "$(dirname "$0")" || exit 1

reps() { echo "1 2 3"; }
cases() { for r in $(reps); do printf "dia64dc_1n48_%s%s " "$1" "$r"; done; }

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

# stdin: rows "case total dft diag sh status"; $1 = label, $2 = tag (g/o/c).
# Emits "MEAN tag phase mean sd" on stderr.
stats() {
  awk -v label="$1" -v tag="$2" '
    $6=="OK" { n++; for (i=2;i<=5;i++){ s[i]+=$i; q[i]+=$i*$i; if(n==1||$i<lo[i])lo[i]=$i; if(n==1||$i>hi[i])hi[i]=$i } }
    $6!="OK" { bad = bad " " $1 }
    END {
      if (n==0) { printf "  %-26s no usable runs:%s\n", label, bad; exit }
      printf "  %-26s (n=%d)%s\n", label, n, (bad=="" ? "" : "   [excluded:" bad " ]")
      printf "    %-16s %10s %8s %9s %9s\n", "phase", "mean", "sd", "min", "max"
      split("2 4 5", idx, " "); split("Total Diagonalization Set_Hamiltonian", nm, " ")
      for (k=1;k<=3;k++) {
        i = idx[k]; m = s[i]/n
        sd = (n>1) ? sqrt((q[i]-n*m*m)/(n-1)) : 0
        printf "    %-16s %10.2f %8.2f %9.2f %9.2f\n", nm[k], m, sd, lo[i], hi[i]
        printf "MEAN %s %s %.4f %.4f\n", tag, nm[k], m, sd > "/dev/stderr"
      }
    }'
}

echo "dia64_dc-lno, 1 node x 48 ranks, 3 repetitions per point"
echo "64 C atoms, C6.0-s2p2d1 -> 832 basis functions, scf.EigenvalueSolver dc-lno"
echo "Gamma-only, 50 SCF max to 1e-10, orderN.HoppingRanges 6.0 Ang"
echo "binary: gpusolver2 merge, commit f5fb5b3; Max_Time column (slowest rank), s"
echo "configs: g = GPU gpusolver + MPS, scf.gemmul8.enable on (default)"
echo "         o = GPU gpusolver + MPS, scf.gemmul8.enable off (plain cuBLAS FP64)"
echo "         c = CPU elpa2, no GPU reserved"
echo "64 clusters over 48 ranks is uneven (16 ranks x 2 clusters, 32 x 1)."
echo

echo "--- per-run ---"
printf "%-20s %10s %10s %10s %10s  %s\n" case total DFT Diag Set_Ham status
collect "$(cases g) $(cases o) $(cases c)" | \
  awk '{printf "%-20s %10s %10s %10s %10s  %s\n", $1,$2,$3,$4,$5,$6}'
echo

echo "--- statistics (mean +- sample SD over 3 runs) ---"
: > /tmp/.dia64dc_means
collect "$(cases g)" | stats "GPU gpusolver, g8-on"  g 2>>/tmp/.dia64dc_means
collect "$(cases o)" | stats "GPU gpusolver, g8-off" o 2>>/tmp/.dia64dc_means
collect "$(cases c)" | stats "CPU elpa2"             c 2>>/tmp/.dia64dc_means
echo

# Ratio of two measured means; its uncertainty is the two relative SDs added in
# quadrature -- quoting the ratio bare would imply a precision neither run has.
echo "--- ratios (mean/mean, +- in quadrature) ---"
awk '
  $1=="MEAN" { m[$2" "$3]=$4; sd[$2" "$3]=$5 }
  function ratio(num, den, txt, ph,   kn, kd, r, rel) {
    kn = num" "ph; kd = den" "ph
    if ((kn in m) && (kd in m) && m[kn]>0 && m[kd]>0) {
      r = m[kn]/m[kd]
      rel = sqrt((sd[kn]/m[kn])^2 + (sd[kd]/m[kd])^2)
      printf "  %-30s %-16s %6.2fx +- %.2f\n", txt, ph, r, r*rel
    }
  }
  END {
    split("Total Diagonalization Set_Hamiltonian", P, " ")
    for (p=1;p<=3;p++) {
      ratio("c", "g", "CPU / GPU(g8-on)  [speedup]", P[p])
      ratio("c", "o", "CPU / GPU(g8-off) [speedup]", P[p])
      ratio("o", "g", "GPU g8-off / g8-on", P[p])
    }
  }' /tmp/.dia64dc_means
rm -f /tmp/.dia64dc_means
echo

# The decisive section.  If GPU utilisation never rises and device memory stays
# at the MPS baseline, DC-LNO never cleared its dimension >= 800 gate and the
# two GPU configs ran the same CPU eigensolver as elpa2.
echo "--- GPU engagement evidence (g/o only) ---"
echo "  dispatch banner printed by Divide_Conquer_LNO.c, then peak GPU sample:"
for cfg in g o; do
  for c in $(cases $cfg); do
    banner=$(grep -hm1 "<DC-LNO>" "${c}/${c}.joblog" 2>/dev/null | sed 's/^ *//')
    peak=$(grep -ohE '^[0-9]+ %, [0-9]+ MiB' "${c}/${c}.smi" 2>/dev/null | sort -t, -k2 -rn | head -1)
    util=$(grep -ohE '^[0-9]+ %' "${c}/${c}.smi" 2>/dev/null | tr -d ' %' | sort -rn | head -1)
    printf "  %-20s peak %-16s max util %s%%\n" "$c" "${peak:-(no .smi)}" "${util:--}"
    [ -n "$banner" ] && printf "      %s\n" "$banner"
  done
done
echo

echo "--- node each run landed on (slow: bnode013, bnode033) ---"
for cfg in g o c; do
  for c in $(cases $cfg); do
    nl=$(awk '/^=== nodes ===/{f=1;next} /^=== /{f=0} f&&NF==1{printf "%s ",$1}' "${c}/${c}.env" 2>/dev/null)
    flag=""; case "$nl" in *bnode013*|*bnode033*) flag="  <-- SLOW NODE";; esac
    printf "  %-20s %s%s\n" "$c" "${nl:-(no .env)}" "$flag"
  done
done
echo

echo "--- Utot and SCF count ---"
for cfg in g o c; do
  for c in $(cases $cfg); do
    u=$(grep -E "Utot\." "${c}/${c}.out" 2>/dev/null | tail -1 | awk '{print $NF}')
    n=$(grep -cE "^ *SCF=" "${c}/${c}.std" 2>/dev/null)
    printf "  %-20s %-24s SCF=%s\n" "$c" "${u:-(no .out)}" "${n:--}"
  done
done
echo

# |Maximum force| = max over atoms of sqrt(fx^2+fy^2+fz^2), recomputed at 12 dp
# from the coordinates.forces block.  MD.Type=nomd, so there is no MD table in
# the .out and no 8-dp printed value to cross-check against -- the recomputed
# figure is the only one available here.
#
# Caveat worth reading before drawing conclusions from this section: the input
# is ideal diamond, every atom on a symmetry site, so the exact forces are
# zero.  What is tabulated is residual roundoff (~1e-11), not a physical force.
# It confirms no path is grossly wrong, but it discriminates far less than it
# does for a distorted cell.  Utot above is the stronger accuracy check.
echo "--- Maximum force (Hartree/Bohr), recomputed from 12-dp forces ---"
: > /tmp/.dia64dc_force
for cfg in g o c; do
  for c in $(cases $cfg); do
    r=$(awk '/<coordinates.forces/{f=1;next} /coordinates.forces>/{f=0} f&&NF>=8{s=$6*$6+$7*$7+$8*$8; if(s>m)m=s} END{if(m>0)printf "%.12f", sqrt(m)}' "${c}/${c}.out" 2>/dev/null)
    printf "  %-20s %s\n" "$c" "${r:--}"
    echo "$cfg ${r:--}" >> /tmp/.dia64dc_force
  done
done
echo
awk '
  { if (!($1 in v)) v[$1]=$2; else if (v[$1]!=$2) v[$1]="REPS-DIFFER" }
  END {
    printf "  on %s | off %s | CPU %s", v["g"], v["o"], v["c"]
    if (v["g"]==v["o"] && v["o"]==v["c"] && v["g"]!="REPS-DIFFER") printf "  -> identical to 1e-12\n"
    else printf "  -> off-on = %+.1e, CPU-on = %+.1e\n", v["o"]-v["g"], v["c"]-v["g"]
  }' /tmp/.dia64dc_force
rm -f /tmp/.dia64dc_force
echo

echo "--- peak host memory, node 0 (GiB) ---"
for cfg in g o; do
  for c in $(cases $cfg); do
    printf "  %-20s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.smi 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
  done
done
for c in $(cases c); do
  printf "  %-20s %s\n" "$c" "$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' ${c}/${c}.mem 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)"
done
