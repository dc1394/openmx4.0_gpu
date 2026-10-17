#!/usr/bin/env bash
# Four GPUs by default; --local needs two, --single needs one GPU/rank.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/tests/hip_test_build.sh"
build=${OPENMX_GPU_TEST_BUILD:-$(mktemp -d)}
mkdir -p "$build"
mpicc=${MPICC:-mpicc}
includes=(-I"$root/source" -I"$rocm/include")
for directory in $("$mpicc" --showme:incdirs); do includes+=(-I"$directory"); done
read -r -a mpi_link <<< "$("$mpicc" --showme:link)"
flags=(-O2 -std=c11 -D_GNU_SOURCE -D__HIP_PLATFORM_AMD__ -fopenmp --offload-arch="$gpu_arch")
objects=()
for name in openmx_gpu_task openmx_gpu_workers openmx_gpu_collective; do
  "$cc" "${flags[@]}" "${includes[@]}" -c "$root/source/$name.c" -o "$build/$name.o"
  objects+=("$build/$name.o")
done
"$cc" "${flags[@]}" "${includes[@]}" -c "$root/tests/gpu_task_mpi_check.c" -o "$build/fixture.o"
"$cc" "${flags[@]}" "$build/fixture.o" "${objects[@]}" "${mpi_link[@]}" \
  -L"$rocm/lib" -lamdhip64 -lpthread -lm -Wl,--wrap=hipMalloc -o "$build/gpu_task_mpi_check"
if [[ ${OPENMX_GPU_TEST_COMPILE_ONLY:-0} != 1 ]]; then
  if [[ ${1:-} == --local || ${1:-} == --single ]]; then
    "${MPIRUN:-mpirun}" -np 1 "$build/gpu_task_mpi_check" "$1"
  else
    "${MPIRUN:-mpirun}" -np 2 "$build/gpu_task_mpi_check" --logical-nodes
  fi
fi
