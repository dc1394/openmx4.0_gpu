#!/bin/bash
# Assemble siband_search_result.txt: the size-search table required by
# research plan v2.6 sec. 7.3.2 item 8 (max safe size, first failing
# candidate above it, probe/confirmation runtimes, node peak RSS, peak VRAM,
# fallbacks, OOM boundary), for the v3.2.0 campaign case dirs on disk.
# Idempotent: re-run any time; it reads only sic_*/ artifacts.
# Run from work/:  ./report_siband_search.sh > siband_search_result.txt
set -u
cd "$(dirname "$0")" || exit 1

BIN_MD5=$(md5sum openmx | cut -d' ' -f1)
GATE=102

row() {  # $1=case  -> one table line (empty if case dir absent)
  local c=$1 jl smi rss dev tot nscf fb verdict note
  [ -d "$c" ] || return 0
  jl="$c/$c.joblog"; smi="$c/$c.smi"
  if [ ! -s "$jl" ]; then printf "  %-16s (no result yet)\n" "$c"; return 0; fi
  rss=$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' "$smi" 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)
  dev=$(grep -oE '^[0-9]+ %, [0-9]+ MiB' "$smi" 2>/dev/null | awk -F'[ ,]+' '{print $3}' | sort -rn | head -1)
  tot=$(grep -m1 "^TIMING " "$jl" | sed -E 's/.* total=([0-9.]+).*/\1/; t; s/.*INCOMPLETE.*/-/')
  nscf=$(grep -m1 "SCF iterations completed" -A1 "$jl" | tail -1)
  fb=$(grep -ciE "cannot fit|Falling back|<Cluster_DFT_" "$jl" 2>/dev/null)
  local used=${rss%%/*}
  if [ -z "$rss" ]; then verdict="?"
  elif [ "$tot" = "-" ] || [ -z "$tot" ]; then verdict="FAIL-OOM"
  elif [ "$used" -gt "$GATE" ]; then verdict="FAIL-gate"
  elif [ "$fb" -gt 0 ]; then verdict="FAIL-fallback"
  else verdict="PASS"
  fi
  printf "  %-16s %9s %6s %8s %6s %5s   %s\n" \
    "$c" "${rss:-?}" "${dev:+$((dev/1024))G}" "${tot:--}" "${nscf:-?}" "$fb" "$verdict"
}

echo "======================================================================"
echo "Si diamond CLUSTER-solver 1-node size search  (research plan v2.6, step 10 (7.3 gates))"
echo "======================================================================"
echo "goal      : per mode (col band / NC band), the largest ladder candidate"
echo "            that runs SAFELY at 1 node x 48 ranks x 1 H100 (MPS on),"
echo "            then 25-SCF confirmation with cuBLAS and with GEMMul8."
echo "gate      : node peak host RSS <= ${GATE} GiB (80% of 128 GB); exit 0; no"
echo "            'cannot fit' GPU fallback.  Probes are single samples:"
echo "            GO/NO-GO only, never a timing ranking."
echo "binary    : work/openmx md5 ${BIN_MD5}"
echo "            (openmx cd5f0d5 + bundled GEMMul8 v3.2.0/833e576,"
echo "            NVHPC 26.5 / CUDA 13.2, sm_90; built $(date -r openmx '+%Y-%m-%d %H:%M'))"
echo "deck      : from ../sidia.dat -- Si7.0-s2p2d1 (13 basis/atom), GGA-PBE,"
echo "            200 Ry, Band solver, Kgrid 2x2x2 (8 computed k), rmm-diisk"
echo "            history 7, scf.criterion 1e-13, MD.Type Opt / MD.maxIter 1;"
echo "            probes truncate to scf.maxIter 3, confirmations run 25."
echo "ladder    : see siband_candidates.txt (frozen before measurement)"
echo
echo "$(cat siband_candidates.txt)"
echo
echo "--- probe / confirmation table -------------------------------------"
echo "  case              RSS(GiB)  devPk    total    SCF  cfit  verdict"
echo "  [col cluster: n = 13*N real]"
for a in 216 250 288 300 360 384 432 512; do row "sic_col${a}_p"; done
for a in 300 360; do row "sic_col${a}_p8"; done
for a in 300 360; do row "sic_col${a}_o"; row "sic_col${a}_g"; done
echo "  [NC cluster: 2n = 26*N complex]"
for a in 216 250 288 300 360 384; do row "sic_nc${a}_p"; done
for a in 300; do row "sic_nc${a}_p8"; done
for a in 300; do row "sic_nc${a}_o"; row "sic_nc${a}_g"; done
echo
echo "  RSS = node peak host 'used' from the 5-s sampler (rank-level RSS is"
echo "  not sampled; the gate is applied at node level).  devPk = peak GPU"
echo "  memory.used.  total = wall seconds, Max_Time column, single run."
echo "  cfit = count of 'cannot fit' dense-GPU fallback banners (must be 0)."
echo "  FAIL-OOM = a rank was SIGKILLed by the host OOM killer mid-run."
echo "  FAIL-gate = run completed but exceeded the ${GATE}-GiB safety gate."
echo
echo "--- GPU engagement (chosen candidates) ------------------------------"
for c in sic_col360_p sic_col360_p8 sic_nc300_p sic_nc300_p8; do
  [ -s "$c/$c.joblog" ] || continue
  b=$(grep -m1 -E "<Set_Hamiltonian> GPU device" "$c/$c.joblog" | sed 's/^ *//')
  printf "  %-16s %s\n" "$c" "${b:-?}"
done
echo
echo "--- Utot cross-checks ------------------------------------------------"
echo "  (3-SCF probes, unconverged by construction: byte-comparison across"
echo "   binaries / backends at identical iteration counts)"
for a in 216 250 288 300 360; do
  for m in col nc; do
    new="sic_${m}${a}_p/sic_${m}${a}_p.out"; old="siband_v311_superseded/sic_${m}${a}_p/sic_${m}${a}_p.out"
    [ -s "$new" ] && [ -s "$old" ] || continue
    un=$(grep -E "Utot\." "$new" | tail -1 | awk '{print $NF}')
    uo=$(grep -E "Utot\." "$old" | tail -1 | awk '{print $NF}')
    same=$([ "$un" = "$uo" ] && echo "IDENTICAL" || echo "DIFFER")
    printf "  %-10s v3.2.0-build %-22s v3.1.1-build %-22s %s\n" "${m}${a}" "$un" "$uo" "$same"
  done
done
for a in 300 360; do
  o="sic_col${a}_p/sic_col${a}_p.out"; g="sic_col${a}_p8/sic_col${a}_p8.out"
  [ -s "$o" ] && [ -s "$g" ] || continue
  printf "  col%-7s cuBLAS %-28s GEMMul8 %-28s\n" "$a" \
    "$(grep -E 'Utot\.' "$o" | tail -1 | awk '{print $NF}')" \
    "$(grep -E 'Utot\.' "$g" | tail -1 | awk '{print $NF}')"
done
for a in 216 250; do
  o="sic_nc${a}_p/sic_nc${a}_p.out"; g="sic_nc${a}_p8/sic_nc${a}_p8.out"
  [ -s "$o" ] && [ -s "$g" ] || continue
  printf "  nc%-8s cuBLAS %-28s GEMMul8 %-28s\n" "$a" \
    "$(grep -E 'Utot\.' "$o" | tail -1 | awk '{print $NF}')" \
    "$(grep -E 'Utot\.' "$g" | tail -1 | awk '{print $NF}')"
done
echo
echo "--- node landings (slow: bnode013, bnode033) -------------------------"
for d in sic_*_p sic_*_p8 sic_*_o sic_*_g; do
  [ -s "$d/$d.env" ] || continue
  nl=$(awk '/^=== nodes ===/{f=1;next} /^=== /{f=0} f&&NF==1{printf "%s ",$1}' "$d/$d.env")
  flag=""; case "$nl" in *bnode013*|*bnode033*) flag="  <-- SLOW NODE (verdicts unaffected: GO/NO-GO only)";; esac
  printf "  %-16s %s%s\n" "$d" "$nl" "$flag"
done
