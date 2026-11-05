#!/usr/bin/env bash
# CPU-only ArmPL comparison, sharing the validated LP64 ELPA/ScaLAPACK archives.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="/usr/lib64/openmpi/bin:$HOME/.local/opt/openmx-autotools/bin:$HOME/.local/bin:$PATH"
export OMPI_CC=/usr/bin/gcc14-gcc OMPI_CXX=/usr/bin/gcc14-g++ OMPI_FC=/usr/bin/gcc14-gfortran
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
jobs=${OPENMX_LOCAL_BUILD_JOBS:-12}
armpl_prefix=${OPENMX_LOCAL_ARMPL_ROOT:-$HOME/.local/opt/armpl-26.07/armpl_26.07_gcc}
blas="$armpl_prefix/lib/libarmpl_lp64.so"
test -f "$blas"
if [[ ! -f build/cpu-sve256/cpu-elpa/lib/libelpa_openmp.a ]]; then
  bash build-sve256-local.sh
fi
arch_flags='-mcpu=neoverse-v1 -msve-vector-bits=scalable -fno-fast-math -ffp-contract=off'
cmake -S . -B build/cpu-sve256-armpl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENMX_GPU_BACKEND=CPU -DOPENMX_CPU_COMPILER=GCC \
  -DOPENMX_CPU_MPI=OPENMPI -DOPENMX_CPU_MPI_ROOT=/usr/lib64/openmpi \
  -DOPENMX_CPU_LINALG=CUSTOM \
  -DOPENMX_CPU_LINALG_LIBRARIES="$PWD/build/cpu-sve256/cpu-scalapack/lib/libscalapack.a;$blas" \
  -DOPENMX_CPU_ELPA_ROOT="$PWD/build/cpu-sve256/cpu-elpa" \
  -DOPENMX_CPU_SCALAPACK_BLOCK_SIZE="${OPENMX_LOCAL_SCALAPACK_BLOCK_SIZE:-32}" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc14-gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/gcc14-g++ \
  -DCMAKE_Fortran_COMPILER=/usr/bin/gcc14-gfortran \
  -DCMAKE_C_FLAGS="$arch_flags" -DCMAKE_CXX_FLAGS="$arch_flags" \
  -DCMAKE_Fortran_FLAGS="$arch_flags" \
  -DOPENMX_HOST_SIMD_TUNE=neoverse-v1 \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/opt/openmx-sve256-armpl" \
  -DOPENMX_COPY_TO_WORK=OFF -DOPENMX_BUILD_JOBS="$jobs"
cmake --build build/cpu-sve256-armpl --parallel "$jobs" \
  --target openmx openmx_cpu_numerics
cmake --install build/cpu-sve256-armpl
