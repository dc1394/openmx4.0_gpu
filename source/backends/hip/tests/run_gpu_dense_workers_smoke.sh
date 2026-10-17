#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/tests/hip_test_build.sh"
build=${OPENMX_GPU_TEST_BUILD:-$(mktemp -d)}
mkdir -p "$build"
includes=(-I"$root/source" -I"$rocm/include")
if [[ -n ${MPI_INCLUDE:-} ]]; then
  includes+=(-I"$MPI_INCLUDE")
else
  for directory in $(mpicc --showme:incdirs); do includes+=(-I"$directory"); done
fi
for name in openmx_gpu_workers; do
  "$cc" -O2 -std=c11 -fopenmp "${includes[@]}" -c "$root/source/$name.c" -o "$build/$name.o"
done
"$cc" -O2 -std=c11 -fopenmp "${includes[@]}" -c "$root/tests/gpu_workers_test_hooks.c" -o "$build/hooks.o"
"$hipcc" -O2 -fopenmp --offload-arch="$gpu_arch" -x hip "${includes[@]}" \
  -c "$root/source/openmx_gpu_views.c" -o "$build/openmx_gpu_views.o"
for name in openmx_gpu_transform openmx_gpu_dm; do
  "$hipcc" -O3 -std=c++17 --offload-arch="$gpu_arch" "${includes[@]}" -c "$root/source/$name.cu" -o "$build/$name.o"
done
"$hipcc" -O2 -fopenmp -std=c++17 --offload-arch="$gpu_arch" "${includes[@]}" \
  -c "$root/tests/gpu_dense_workers_smoke.cu" -o "$build/fixture.o"
"$hipcc" -fopenmp --offload-arch="$gpu_arch" "$build/fixture.o" "$build/hooks.o" \
  "$build/openmx_gpu_workers.o" "$build/openmx_gpu_views.o" "$build/openmx_gpu_transform.o" "$build/openmx_gpu_dm.o" \
  -lpthread -o "$build/gpu_dense_workers_smoke"
if [[ ${OPENMX_GPU_TEST_COMPILE_ONLY:-0} != 1 ]]; then "$build/gpu_dense_workers_smoke"; fi
