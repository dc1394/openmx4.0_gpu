#!/usr/bin/env bash
# Reproducible CPU-only build for this Graviton3 / Neoverse V1 SVE256 host.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="/usr/lib64/openmpi/bin:$HOME/.local/opt/openmx-autotools/bin:$HOME/.local/bin:$PATH"
export OMPI_CC=/usr/bin/gcc14-gcc OMPI_CXX=/usr/bin/gcc14-g++ OMPI_FC=/usr/bin/gcc14-gfortran
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
jobs=${OPENMX_LOCAL_BUILD_JOBS:-12}
blas_src="$PWD/build/deps/OpenBLAS-0.3.34"
blas_prefix="$HOME/.local/opt/openblas-0.3.34-sve256"
mkdir -p build/deps build/sve256-logs
if [[ ! -d "$blas_src/.git" ]]; then
  git clone --depth 1 --branch v0.3.34 https://github.com/OpenMathLib/OpenBLAS.git "$blas_src"
fi
[[ "$(git -C "$blas_src" rev-parse HEAD)" == e0166008be8e466242aa76b2ff75ce3f0fbf574a ]]
blas_args=(TARGET=NEOVERSEV1 DYNAMIC_ARCH=0 USE_THREAD=1 USE_OPENMP=1 NUM_THREADS=32
  CC=/usr/bin/gcc14-gcc FC=/usr/bin/gcc14-gfortran NO_AFFINITY=1)
make -C "$blas_src" -j"$jobs" "${blas_args[@]}" >build/sve256-logs/openblas-build.log 2>&1
make -C "$blas_src" "${blas_args[@]}" PREFIX="$blas_prefix" install >build/sve256-logs/openblas-install.log 2>&1
# GCC 14 fixed-width SVE SLP miscompiled the VPS metadata callback check.
# VLA auto-vectorization passes it and still executes with this host's 256-bit
# VL. Explicit host SIMD and ELPA kernels retain their own fixed SVE256 flags.
arch_flags='-mcpu=neoverse-v1 -msve-vector-bits=scalable -fno-fast-math -ffp-contract=off'
cmake -S . -B build/cpu-sve256 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENMX_GPU_BACKEND=CPU -DOPENMX_CPU_COMPILER=GCC \
  -DOPENMX_CPU_MPI=OPENMPI -DOPENMX_CPU_MPI_ROOT=/usr/lib64/openmpi \
  -DOPENMX_CPU_LINALG=OPENBLAS -DOPENMX_CPU_OPENBLAS_ROOT="$blas_prefix" \
  -DOPENMX_CPU_ELPA_KERNELS=sve256 \
  -DOPENMX_CPU_SCALAPACK_BLOCK_SIZE="${OPENMX_LOCAL_SCALAPACK_BLOCK_SIZE:-32}" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc14-gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/gcc14-g++ \
  -DCMAKE_Fortran_COMPILER=/usr/bin/gcc14-gfortran \
  -DCMAKE_C_FLAGS="$arch_flags" -DCMAKE_CXX_FLAGS="$arch_flags" \
  -DCMAKE_Fortran_FLAGS="$arch_flags" \
  -DOPENMX_HOST_SIMD_TUNE=neoverse-v1 \
  -DOPENMX_CORE_BUILD_BENCHMARKS=ON \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/opt/openmx-sve256" \
  -DOPENMX_COPY_TO_WORK=OFF -DOPENMX_BUILD_JOBS="$jobs"
cmake --build build/cpu-sve256 --parallel "$jobs"
cmake --install build/cpu-sve256
