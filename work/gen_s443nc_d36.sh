#!/bin/bash
# Generate the 12 sidia443_nc_cluster gpusolver2 cases at the ORIGINAL
# campaign's 36 ranks/node: s443nc_{1,2,3,4}n_d{1,2,3}.  They join the
# existing g/o/c namespace so summarize_s443nc_reps.sh can aggregate them
# against the already-measured 3-rep g/o/c data.  Binary = gpusolver2 merge
# (commit f5fb5b3).  The 48-rank attempt OOMed on 1 node (117-119/124 GiB),
# hence this 36-rank campaign.  Every derived file is asserted before use.
set -u
cd "$(dirname "$0")" || exit 1

fail=0
say() { printf '%s\n' "$*"; }
need() {
  if [ "$2" = "$3" ]; then say "  PASS $1"; else say "  FAIL $1: expected [$2] got [$3]"; fail=1; fi
}

for n in 1 2 3 4; do
  tpl=s443nc_${n}n_g1
  for r in 1 2 3; do
    case=s443nc_${n}n_d${r}
    mkdir -p "$case"

    sed -e "s/${tpl}/${case}/g" \
        -e "s/^#PBS -N s443_${n}g1\$/#PBS -N s443_${n}d${r}/" \
        -e "s/elapstim_req=00:40:00/elapstim_req=01:00:00/" \
        -e "s/GPU path (gemmul8 on)/GPU path (gpusolver2)/" \
        -e "s/MPS on, gemmul8 on/MPS on, gpusolver2/" \
        -e "s/{gpusolver+gemmul8-on, gpusolver+gemmul8-off, elpa2}/{gpusolver2, gpusolver, elpa2}/" \
        -e "s/round-3 (commit e227507)/gpusolver2-merge (commit f5fb5b3)/" \
        "${tpl}/${tpl}.sh" > "${case}/${case}.sh"

    sed -e "s/${tpl}/${case}/g" \
        -e "s/^\(scf.eigen.lib  *\)gpusolver\$/\1gpusolver2/" \
        "${tpl}/${tpl}.dat" > "${case}/${case}.dat"

    cp "${tpl}/mps_node.sh" "${case}/mps_node.sh"

    say "== ${case}"
    need "System.Name" "$case" "$(awk '$1=="System.Name"{print $2}' "${case}/${case}.dat")"
    need "scf.eigen.lib" "gpusolver2" "$(awk '$1=="scf.eigen.lib"{print $2}' "${case}/${case}.dat")"
    need "no gemmul8 line" "0" "$(grep -c "scf.gemmul8" "${case}/${case}.dat")"
    need "CASE var" "CASE=${case}" "$(grep -m1 '^CASE=' "${case}/${case}.sh")"
    need "PPN" "PPN=36" "$(grep -m1 '^PPN=' "${case}/${case}.sh")"
    need "-b nodes" "#PBS -b ${n}" "$(grep -m1 '^#PBS -b ' "${case}/${case}.sh")"
    need "job name" "#PBS -N s443_${n}d${r}" "$(grep -m1 '^#PBS -N ' "${case}/${case}.sh")"
    need "joblog" "#PBS -o ${case}.joblog" "$(grep -m1 '^#PBS -o ' "${case}/${case}.sh")"
    need "elapstim" "#PBS -l elapstim_req=01:00:00" "$(grep -m1 '^#PBS -l elapstim' "${case}/${case}.sh")"
    need "no stale refs" "0" "$(grep -c "${tpl}" "${case}/${case}.sh" "${case}/${case}.dat" | awk -F: '{s+=$2} END{print s}')"
    need "mps_node.sh" "yes" "$([ -s "${case}/mps_node.sh" ] && echo yes || echo no)"
    bash -n "${case}/${case}.sh" && say "  PASS bash -n" || { say "  FAIL bash -n"; fail=1; }
  done
done

[ $fail -eq 0 ] && say "ALL 12 D-CASES GENERATED AND ASSERTED OK" || { say "GENERATION HAD FAILURES"; exit 1; }
