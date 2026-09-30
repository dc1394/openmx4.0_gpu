#!/bin/sh
set -eu

# CUDA_HOME and CUDA_MATH_ROOT can select an older toolkit for fallback checks.
# Example: CUDA_HOME=/usr/local/cuda-13.2 tests/run_cusolver_smoke.sh 257
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda-13.4}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
sdk_root=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/openmx-cusolver-test.XXXXXX")
trap 'rm -rf "$test_tmp"' EXIT HUP INT TERM
unset LD_LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
# The SDK's mpi/ directory is a selector, not an MPI installation. Resolve
# its real header directories through the selected wrapper.
comm_root=${NVCOMPILER_COMM_LIBS_HOME:-$sdk_root/comm_libs/13.3}
export NVCOMPILER_COMM_LIBS_HOME=$comm_root
mpi_includes=$("$sdk_root/comm_libs/mpi/bin/mpicc" --showme:incdirs)
mpi_flags=
for inc in $mpi_includes; do mpi_flags="$mpi_flags -I$inc"; done

${CC:-cc} -O2 -std=c11 -I"$test_root/source" -I"$math_root/include" -I"$cuda_root/include" \
    "$test_root/tests/cusolver_smoke.c" "$test_root/source/openmx_cusolver_compat.c" \
    -L"$math_root/lib64" -L"$cuda_root/lib64" \
    -Wl,--disable-new-dtags,-rpath,"$math_root/lib64",-rpath,"$cuda_root/lib64" \
    -lcusolver -lcudart -lm -o "$test_tmp/api"

NVHPC_CUDA_HOME="$cuda_root" "$sdk_root/compilers/bin/nvc" -O2 -std=c11 -acc -mp \
    -I"$test_root/source" -I"$math_root/include" -I"$cuda_root/include" \
    $mpi_flags \
    "$test_root/tests/gpusolver_wrapper_smoke.c" "$test_root/source/gpusolver_Syevdx.c" \
    "$test_root/source/openmx_cusolver_compat.c" \
    -L"$math_root/lib64" -L"$cuda_root/lib64" \
    -Wl,--disable-new-dtags,-rpath,"$math_root/lib64",-rpath,"$cuda_root/lib64" \
    -lcusolver -lcudart -lm -o "$test_tmp/wrappers"

for mode in auto two-stage legacy; do
    export OPENMX_CUSOLVER_ALGORITHM=$mode OPENMX_CUSOLVER_VERBOSE=1
    if [ "$mode" = legacy ]; then
        export OPENMX_CUSOLVER_EMULATION=0
    else
        export OPENMX_CUSOLVER_EMULATION=1
    fi
    "$test_tmp/api" "${1:-257}"
    "$test_tmp/wrappers"
done
