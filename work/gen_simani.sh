#!/bin/bash
# Generate run_manifest.json validation runs (run_manifest_design.md sec.4)
# by cloning existing sip_/siacc_ run dirs: same physics decks, same job
# harness (MPS, watchdog, TIMING extraction), only maxIter / names / env
# differ.  Point spec:   <tag>:<srcdir>:<maxiter|keep>:<ENV=VAL|->
# Produces work/simani_<tag>/ ready for qsub.
#   ./gen_simani.sh bg:sip_bcol300_g1:3:- fb1:sip_bcol300_g1:3:OPENMX_BAND_GPU_DIAG=0 ...
set -u
cd "$(dirname "$0")" || exit 1

fail() { echo "FAIL: $*" >&2; exit 1; }

for spec in "$@"; do
  IFS=: read -r tag src maxiter envkv <<<"$spec"
  [ -n "$tag" ] && [ -n "$src" ] && [ -n "$maxiter" ] && [ -n "$envkv" ] || fail "bad spec '$spec'"
  dst="simani_${tag}"
  [ -d "$src" ] || fail "$spec: source dir $src not found"
  [ -f "$src/$src.dat" ] && [ -f "$src/$src.sh" ] || fail "$spec: $src lacks deck or script"
  rm -rf "$dst"; mkdir "$dst" || fail "mkdir $dst"

  sed "s/${src}/${dst}/g" "$src/$src.dat" > "$dst/$dst.dat"
  sed "s/${src}/${dst}/g" "$src/$src.sh"  > "$dst/$dst.sh"
  [ -f "$src/mps_node.sh" ] && cp -p "$src/mps_node.sh" "$dst/"

  # pin the SCF count (or keep the source count for noise reps)
  if [ "$maxiter" != keep ]; then
    sed -i "s/^scf\.maxIter\([[:space:]]\{1,\}\)[0-9]\{1,\}/scf.maxIter\1${maxiter}/" "$dst/$dst.dat"
    sed -i "s/^#PBS -l elapstim_req=.*/#PBS -l elapstim_req=00:20:00/" "$dst/$dst.sh"
  fi
  sed -i "s/^#PBS -N .*/#PBS -N pm${tag}/" "$dst/$dst.sh"

  # optional environment override, exported and forwarded through mpirun
  if [ "$envkv" != "-" ]; then
    name=${envkv%%=*}
    grep -q '^MPS_X=' "$dst/$dst.sh" || fail "$spec: env requested but no MPS_X line in $src.sh"
    sed -i "s/^ulimit -c 0/ulimit -c 0\nexport ${envkv}/" "$dst/$dst.sh"
    sed -i "s/^MPS_X=\"\(.*\)\"/MPS_X=\"\1 -x ${name}\"/" "$dst/$dst.sh"
  fi

  # ---- assertions ----
  n=$(grep -c "$dst" "$dst/$dst.sh"); [ "$n" -ge 2 ] || fail "$dst: name rewrite looks wrong ($n hits)"
  grep -q "$src" "$dst/$dst.sh" && fail "$dst: stale source name in script"
  grep -q "$src" "$dst/$dst.dat" && fail "$dst: stale source name in deck"
  grep -q "^System.Name[[:space:]]*${dst}\$" <(awk '{$1=$1};1' "$dst/$dst.dat" | grep '^System.Name') || \
    grep -q "System.Name.*${dst}" "$dst/$dst.dat" || fail "$dst: System.Name not rewritten"
  if [ "$maxiter" != keep ]; then
    grep -qE "^scf\.maxIter[[:space:]]+${maxiter}\b" "$dst/$dst.dat" || fail "$dst: maxIter not ${maxiter}"
    grep -q "elapstim_req=00:20:00" "$dst/$dst.sh" || fail "$dst: walltime not tightened"
  fi
  if [ "$envkv" != "-" ]; then
    grep -q "^export ${envkv}\$" "$dst/$dst.sh" || fail "$dst: export ${envkv} missing"
    grep -qE "^MPS_X=.* -x ${name}\"" "$dst/$dst.sh" || fail "$dst: -x ${name} missing from MPS_X"
  fi
  chmod +x "$dst/$dst.sh"
  echo "OK  $dst  (from $src, maxIter=${maxiter}, env=${envkv})"
done
