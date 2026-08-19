#!/bin/bash
# Validate <case>.manifest.json against the joblog/deck the way the
# campaign summarizers scrape them (run_manifest_design.md sec.4-1) plus
# the three deliberate-fallback scenarios (sec.4-3).
# Run from work/ after the simani_* wave:  ./validate_manifest.sh
set -u
cd "$(dirname "$0")" || exit 1

PASS=0; FAILN=0
ok()   { PASS=$((PASS+1));  printf "  PASS  %s\n" "$1"; }
bad()  { FAILN=$((FAILN+1)); printf "  FAIL  %s\n" "$1"; }
chk()  { # chk <desc> <condition-result 0/1>
  if [ "$2" = 0 ]; then ok "$1"; else bad "$1"; fi
}

# jget <file> <dotted.path>  -> value ("MISSING" if absent)
jget() {
  python3 - "$1" "$2" <<'EOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    for k in sys.argv[2].split('.'):
        d=d[k]
    if isinstance(d,bool): print("true" if d else "false")
    else: print(d)
except Exception:
    print("MISSING")
EOF
}

num_eq()  { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0==b+0)}'; }
num_gt0() { awk -v a="$1" 'BEGIN{exit !(a+0>0)}'; }
num_near(){ awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; m=(b<0?-b:b); exit !(d<=t || (m>0 && d/m<=t))}'; }

check_case() { # <case> <expectations...> as key=val pairs
  local c=$1; shift
  local mf="$c/$c.manifest.json" jl="$c/$c.joblog" dk="$c/$c.dat"
  echo "[$c]"
  if [ ! -s "$mf" ]; then bad "manifest file exists"; return; fi
  ok "manifest file exists"
  python3 -c "import json;json.load(open('$mf'))" 2>/dev/null && ok "JSON parses" || { bad "JSON parses"; return; }

  # universal checks -------------------------------------------------
  local tot scf jscf sha md5m md5b
  tot=$(jget "$mf" wall.total_s)
  jtot=$(grep -oE "TIMING $c total=[0-9.]+" "$jl" | sed 's/.*total=//')
  [ -n "$jtot" ] && { num_near "$tot" "$jtot" 0.02; chk "wall.total_s ($tot) ~ TIMING ($jtot)" $?; }

  scf=$(jget "$mf" wall.scf_iters)
  jscf=$(awk '/=== SCF iterations completed/{getline; print; exit}' "$jl")
  [ -n "$jscf" ] && { num_eq "$scf" "$jscf"; chk "scf_iters ($scf) == joblog ($jscf)" $?; }

  sha=$(jget "$mf" input.sha256)
  dsha=$(sha256sum "$dk" | awk '{print $1}')
  [ "$sha" = "$dsha" ]; chk "input.sha256 matches deck" $?

  md5m=$(jget "$mf" build.md5); md5b=$(md5sum openmx | awk '{print $1}')
  [ "$md5m" = "$md5b" ]; chk "build.md5 matches work/openmx" $?

  # diag iter counters sum to scf_iters (one B09 line per SCF iteration)
  local dg dc
  dg=$(jget "$mf" counters.diag_gpu_iters); dc=$(jget "$mf" counters.diag_cpu_iters)
  num_eq "$((dg+dc))" "$scf"; chk "diag_gpu+cpu iters ($dg+$dc) == scf_iters" $?

  # Set_Hamiltonian plan banner vs manifest
  local br
  br=$(grep -oE "<Set_Hamiltonian> GPU device [0-9]+: [0-9]+ Hamiltonian rank" "$jl" | head -1 | grep -oE "[0-9]+ Hamiltonian" | awk '{print $1}')
  if [ -n "$br" ]; then
    m=$(jget "$mf" phases.set_hamiltonian.gpu_ranks)
    num_eq "$m" "$br"; chk "setham gpu_ranks ($m) == banner ($br)" $?
  fi

  # per-case expectations --------------------------------------------
  local kv k v
  for kv in "$@"; do
    k=${kv%%=*}; v=${kv#*=}
    m=$(jget "$mf" "$k")
    case $v in
      +) num_gt0 "$m";            chk "$k ($m) > 0" $?;;
      0) num_eq  "$m" 0;          chk "$k ($m) == 0" $?;;
      *) [ "$m" = "$v" ];         chk "$k ($m) == $v" $?;;
    esac
  done
  echo
}

echo "=== manifest validation (design sec.4-1 unit + sec.4-3 fallbacks) ==="
echo "binary: $(md5sum openmx)"
echo

check_case simani_bg \
  dense_solver.path=gpusolver-gpu-dense counters.dense_gpu_solves=+ \
  counters.dense_cpu_solves=0 gemmul8.enabled=true gemmul8.z_calls=+ \
  counters.kowner_ranks=+ phases.set_hamiltonian.rank_gpu_iters=+ \
  layout.gpu_ranks=48 dense_solver.demoted_elpa2=false

check_case simani_bo \
  dense_solver.path=gpusolver-gpu-dense counters.dense_gpu_solves=+ \
  gemmul8.enabled=false gemmul8.z_calls=0 gemmul8.input_off_calls=+

check_case simani_nc \
  dense_solver.path=gpusolver-gpu-dense counters.dense_gpu_solves=+ \
  counters.dense_cpu_solves=0 gemmul8.enabled=true gemmul8.z_calls=+ \
  counters.kowner_ranks=+

check_case simani_cc \
  dense_solver.path=gpusolver-gpu-dense counters.dense_gpu_solves=+ \
  counters.dense_cpu_solves=0 gemmul8.enabled=false gemmul8.input_off_calls=+ \
  counters.dm_gpu_calls=+

check_case simani_cn \
  dense_solver.path=gpusolver-gpu-dense counters.dense_gpu_solves=+ \
  counters.dense_cpu_solves=0 gemmul8.enabled=true gemmul8.z_calls=+

check_case simani_cpu \
  input.eigen_lib=elpa2 dense_solver.path=elpa2 \
  counters.dense_cpu_solves=+ counters.dense_gpu_solves=0 \
  counters.gpu_ranks=0 phases.set_hamiltonian.rank_gpu_iters=0 \
  phases.set_hamiltonian.rank_cpu_iters=+ gemmul8.z_calls=0 \
  counters.total_energy_gpu_calls=0

check_case simani_fb1 \
  dense_solver.path=gpusolver-cpu-fallback counters.dense_fb_env_off=1 \
  counters.dense_gpu_solves=0 counters.dense_cpu_solves=+

check_case simani_fb2 \
  dense_solver.path=gpusolver-cpu-fallback counters.dense_fb_mem_events=+ \
  counters.dense_gpu_solves=0 counters.dense_cpu_solves=+

check_case simani_fb3 \
  dense_solver.small_system_cpu_diag=true counters.dense_gpu_solves=0

echo "=== noise reps (25 SCF bnc216 o; pre-manifest mean 253.39 +- 1.74 s) ==="
for r in 1 2 3; do
  c=simani_noise$r
  t=$(grep -oE "TIMING $c total=[0-9.]+" "$c/$c.joblog" 2>/dev/null | sed 's/.*total=//')
  n=$(awk '/^=== nodes ===/{f=1;next} /^=== /{f=0} f&&NF==1{printf "%s ",$1}' "$c/$c.env" 2>/dev/null)
  printf "  %-15s total=%-8s nodes=%s\n" "$c" "${t:-?}" "${n:-?}"
done
echo "  (accept: mean within ~1%% of 253.39 s after slow-node screening;"
echo "   bnode013/bnode033 draws are discarded per site convention)"
echo
echo "=== runtest ==="
grep -E "runtest.result|14/14|OK|failed" runtest_np48.std 2>/dev/null | head -5
tail -20 runtest.result 2>/dev/null | head -20
echo
echo "=== summary: PASS=$PASS FAIL=$FAILN ==="
[ "$FAILN" -eq 0 ]
