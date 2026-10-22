#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
build=${OPENMX_GPU_TEST_BUILD:-$(mktemp -d)}
if [[ -z ${OPENMX_GPU_TEST_BUILD:-} ]]; then trap 'rm -rf "$build"' EXIT; fi
mkdir -p "$build"
"${PYTHON:-python3}" - "$repo" "$build" <<'PY'
import pathlib,sys
repo,build=map(pathlib.Path,sys.argv[1:])
source=(repo/'source/Set_Hamiltonian.c').read_text()
regions=[]
for kind,following in [('Base','static void Set_Hamiltonian_Base_OpenACC('),('Fallback','static int Set_Hamiltonian_Run_OpenACC_MatrixElements(')]:
    end_type=source.index('} SetH_'+kind+'Work;')
    start=source.rfind('typedef struct {',0,end_type)
    end=source.index(following,end_type)
    regions.append(source[start:end])
(build/'setham_workers.inc').write_text('\n'.join(regions))
PY
cuda=${CUDA_HOME:-/usr/local/cuda}
math=${CUDA_MATH_ROOT:-$cuda}
nvc=${NVHPC_CC:-nvc}
nvcxx=${NVHPC_CXX:-nvc++}
arch=${GPU_ARCH:-80}
includes=(-I"$repo/source" -I"$build" -I"$cuda/include" -I"$math/include")
if [[ -n ${MPI_INCLUDE:-} ]]; then includes+=(-I"$MPI_INCLUDE");
else for d in $(mpicc --showme:incdirs); do includes+=(-I"$d"); done; fi
for name in openmx_gpu_workers; do
  "$nvc" -O2 -std=c11 -acc -mp "-gpu=cc$arch" "${includes[@]}" -c "$repo/source/$name.c" -o "$build/$name.o"
done
"${CUDAHOSTCXX:-g++}" -O2 -std=c++17 -fopenmp "${includes[@]}" -c "$repo/source/openmx_gpu_views.cpp" -o "$build/openmx_gpu_views.o"
"$nvc" -O2 -std=c11 -acc -mp "-gpu=cc$arch" "${includes[@]}" -c "$repo/tests/gpu_workers_test_hooks.c" -o "$build/hooks.o"
"$nvc" -O2 -std=c11 "${includes[@]}" -c "$repo/source/openmx_env.c" -o "$build/env.o"
"$nvc" -O2 -std=c11 -Minfo=accel -acc -mp "-gpu=cc$arch" "${includes[@]}" -c "$repo/tests/gpu_setham_workers_smoke.c" -o "$build/fixture.o"
"$nvcxx" -acc -mp "-gpu=cc$arch" "$build/fixture.o" "$build/hooks.o" "$build/env.o" "$build/openmx_gpu_workers.o" "$build/openmx_gpu_views.o" \
  -L"$cuda/lib64" "-Wl,-rpath,$cuda/lib64" -lcudart -lpthread -lm -o "$build/gpu_setham_workers_smoke"
if [[ ${OPENMX_GPU_TEST_COMPILE_ONLY:-0} != 1 ]]; then "$build/gpu_setham_workers_smoke"; fi
