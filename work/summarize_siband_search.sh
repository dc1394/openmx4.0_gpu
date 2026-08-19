#!/bin/bash
# Feasibility verdicts for the Si diamond band-solver size search
# (research plan v2.6 sec. 7.3.2).  For every sib_* case on disk (or those
# named as arguments) print the evidence the plan's gate asks for:
#   - exit status / completion banner / crash line
#   - TIMING line (single sample: GO/NO-GO evidence, never a ranking)
#   - SCF count actually run
#   - Band GPU engagement banner vs "cannot fit" fallback lines
#   - node peak host RSS vs the 102-GiB gate (80% of 128)
#   - peak GPU memory and utilisation
#   - node the run landed on (slow: bnode013, bnode033)
# plus a crude 25-SCF wall-time estimate from 3-SCF probes:
#   T25 <~ Total3 + (22/3)*DFT3   (labelled estimate; DFT3 contains the
#   one-off force step, so this leans high -- useful for walltime setting,
#   never to be reported as a measurement).
#
# Rank-level RSS is not sampled by this harness; the gate is applied at node
# level and that limitation is stated here once rather than hidden.
#
# Run from work/:  ./summarize_siband_search.sh [case ...]
set -u
cd "$(dirname "$0")" || exit 1

RSS_GATE=102

if [ $# -gt 0 ]; then CASES="$*"; else
  CASES=$(ls -d sib_*_[pog] sib_*_p8 2>/dev/null | tr '\n' ' ')
fi

for c in $CASES; do
  [ -d "$c" ] || { echo "== $c: no such case dir"; continue; }
  jl="$c/$c.joblog"; smi="$c/$c.smi"; env="$c/$c.env"; std="$c/$c.std"
  echo "== $c"
  if [ ! -s "$jl" ]; then echo "   (no joblog yet)"; continue; fi

  grep -m1 "=== openmx exit status" "$jl" | sed 's/^/   /'
  t=$(grep -m1 "^TIMING " "$jl")
  echo "   ${t:-TIMING line missing}"

  # SCF count and 25-SCF estimate for 3-SCF probes
  nscf=$(grep -m1 "SCF iterations completed" -A1 "$jl" | tail -1)
  echo "   SCF iterations run: ${nscf:-?}"
  case "$t" in
    *INCOMPLETE*|"") : ;;
    *) tot=$(echo "$t" | sed -E 's/.* total=([0-9.]+).*/\1/')
       dft=$(echo "$t" | sed -E 's/.* dft=([0-9.]+).*/\1/')
       diag=$(echo "$t" | sed -E 's/.* diag=([0-9.]+).*/\1/')
       if [ "${nscf:-0}" = 3 ]; then
         est=$(awk -v T="$tot" -v D="$dft" 'BEGIN{printf "%.0f", T + 22.0/3.0*D}')
         echo "   est. 25-SCF wall time <~ ${est} s  (ESTIMATE from one 3-SCF sample, leans high)"
       fi
       echo "   diag share of dft: $(awk -v a="$diag" -v b="$dft" 'BEGIN{if(b>0)printf "%.0f%%", 100*a/b; else print "?"}')"
       ;;
  esac

  # engagement vs fallback
  eng=$(grep -cE "GPU device [0-9]+: [0-9]+ k-owner rank" "$jl" 2>/dev/null || true)
  fb=$(grep -ciE "cannot fit|disabled by OPENMX_BAND_GPU_DIAG" "$jl" 2>/dev/null || true)
  bl=$(grep -m1 -E "<Band_DFT_(Col|NonCol)> GPU device" "$jl" | sed 's/^ *//')
  echo "   GPU path: engagement lines=${eng}, fallback lines=${fb}"
  [ -n "$bl" ] && echo "     banner: $bl"

  # crash / OOM evidence
  # "Killed  sleep 5" is the harness reaping its own .smi sampler, not a crash
  cr=$(grep -m1 -hE "exited on signal|prterun noticed|Out Of Memory|oom-kill|Killed" "$jl" "$c/$c.err" 2>/dev/null | grep -v "sleep 5")
  [ -n "$cr" ] && echo "   CRASH EVIDENCE: $cr"

  # memory gate
  rss=$(grep -ohE 'host mem used/total \(GiB\): [0-9]+/[0-9]+' "$smi" 2>/dev/null | sed 's#.*: ##' | sort -t/ -k1 -rn | head -1)
  if [ -n "$rss" ]; then
    used=${rss%/*}
    verdict=$(awk -v u="$used" -v g="$RSS_GATE" 'BEGIN{print (u<=g) ? "PASS (<= " g ")" : "FAIL (> " g ")"}')
    echo "   node peak host RSS: ${rss} GiB  -> gate ${verdict}"
  else
    echo "   node peak host RSS: (no .smi)"
  fi
  gpu=$(grep -oE '^[0-9]+ %, [0-9]+ MiB' "$smi" 2>/dev/null | sort -t, -k2 -rn | head -1)
  util=$(grep -oE '^[0-9]+ %' "$smi" 2>/dev/null | tr -d ' %' | sort -rn | head -1)
  echo "   peak GPU: mem sample '${gpu:-?}', max util ${util:-?}%  (H100 80 GiB)"

  node=$(awk '/^=== nodes ===/{f=1;next} /^=== /{f=0} f&&NF==1{printf "%s ",$1}' "$env" 2>/dev/null)
  flag=""; case "$node" in *bnode013*|*bnode033*) flag="  <-- SLOW NODE";; esac
  echo "   node: ${node:-?}${flag}"

  u=$(grep -E "Utot\." "$c/$c.out" 2>/dev/null | tail -1 | awk '{print $NF}')
  [ -n "$u" ] && echo "   Utot: $u Hartree"
  echo
done
echo "note: rank-level RSS is not sampled by this harness; the 102-GiB gate is"
echo "applied to the node total from the 5-s .smi sampler.  Probes are single"
echo "samples: GO/NO-GO only."
