#!/bin/bash
# Generate the 36 sidia443_nc_cluster 48-rank cases:
#   {1,2,3,4} nodes x {d: gpusolver2, g: gpusolver (gemmul8 default on),
#   c: CPU elpa2} x 3 reps, all -npernode 48 (vs the original campaign's 36).
# Derived from the proven s443nc_{n}n_{g,c}1 templates; binary = gpusolver2
# merge (commit f5fb5b3).  Every derived file is asserted before use.
set -u
cd "$(dirname "$0")" || exit 1

fail=0
say() { printf '%s\n' "$*"; }
need() {  # need <label> <expected> <actual>
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

for n in 1 2 3 4; do
  for cfg in d g c; do
    case $cfg in
      d|g) tpl=s443nc_${n}n_g1 ;;
      c)   tpl=s443nc_${n}n_c1 ;;
    esac
    for r in 1 2 3; do
      case=s443nc48_${n}n_${cfg}${r}
      mkdir -p "$case"

      # ---- job script
      sed -e "s/${tpl}/${case}/g" \
          -e "s/^#PBS -N s443_${n}[gc]1\$/#PBS -N s48_${n}${cfg}${r}/" \
          -e "s/^PPN=36\$/PPN=48/" \
          -e "s/-npernode 36/-npernode 48/g" \
          -e "s/{gpusolver+gemmul8-on, gpusolver+gemmul8-off, elpa2}/{gpusolver2, gpusolver, elpa2}/" \
          -e "s/round-3 (commit e227507)/gpusolver2-merge (commit f5fb5b3)/" \
          "${tpl}/${tpl}.sh" > "${case}/${case}.sh"
      if [ "$cfg" = d ]; then
        sed -i -e "s/GPU path (gemmul8 on)/GPU path (gpusolver2)/" \
               -e "s/MPS on, gemmul8 on/MPS on, gpusolver2/" \
               -e "s/elapstim_req=00:40:00/elapstim_req=01:00:00/" \
               "${case}/${case}.sh"
      fi

      # ---- input
      sed -e "s/${tpl}/${case}/g" "${tpl}/${tpl}.dat" > "${case}/${case}.dat"
      if [ "$cfg" = d ]; then
        sed -i "s/^\(scf.eigen.lib  *\)gpusolver\$/\1gpusolver2/" "${case}/${case}.dat"
      fi

      # ---- MPS helper for the GPU configs
      if [ "$cfg" != c ]; then cp "${tpl}/mps_node.sh" "${case}/mps_node.sh"; fi

      # ---- assertions
      say "== ${case}"
      need "System.Name" "$case" "$(awk '$1=="System.Name"{print $2}' "${case}/${case}.dat")"
      case $cfg in
        d) want=gpusolver2 ;;
        g) want=gpusolver ;;
        c) want=elpa2 ;;
      esac
      need "scf.eigen.lib" "$want" "$(awk '$1=="scf.eigen.lib"{print $2}' "${case}/${case}.dat")"
      need "no gemmul8 line" "0" "$(grep -c "scf.gemmul8" "${case}/${case}.dat")"
      need "CASE var" "CASE=${case}" "$(grep -m1 '^CASE=' "${case}/${case}.sh")"
      need "PPN" "PPN=48" "$(grep -m1 '^PPN=' "${case}/${case}.sh")"
      need "-b nodes" "#PBS -b ${n}" "$(grep -m1 '^#PBS -b ' "${case}/${case}.sh")"
      need "job name" "#PBS -N s48_${n}${cfg}${r}" "$(grep -m1 '^#PBS -N ' "${case}/${case}.sh")"
      need "joblog" "#PBS -o ${case}.joblog" "$(grep -m1 '^#PBS -o ' "${case}/${case}.sh")"
      case $cfg in
        d) wt=01:00:00 ;;
        g) wt=00:40:00 ;;
        c) wt=01:30:00 ;;
      esac
      need "elapstim" "#PBS -l elapstim_req=${wt}" "$(grep -m1 '^#PBS -l elapstim' "${case}/${case}.sh")"
      need "no stale refs" "0" "$(grep -c "s443nc_${n}n_" "${case}/${case}.sh" "${case}/${case}.dat" | awk -F: '{s+=$2} END{print s}')"
      if [ "$cfg" != c ]; then
        need "mps_node.sh" "yes" "$([ -s "${case}/mps_node.sh" ] && echo yes || echo no)"
      fi
      bash -n "${case}/${case}.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
    done
  done
done

[ $fail -eq 0 ] && say "ALL 36 CASES GENERATED AND ASSERTED OK" || { say "GENERATION HAD FAILURES"; exit 1; }
