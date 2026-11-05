#!/usr/bin/env bash
# Rebuild this Amazon Linux 2023 / AArch64 CPU-only installation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="/usr/lib64/openmpi/bin:/home/ec2-user/.local/opt/openmx-autotools/bin:/home/ec2-user/.local/bin:$PATH"
cmake -S . -B build/cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENMX_GPU_BACKEND=CPU \
  -DOPENMX_CPU_COMPILER=GCC \
  -DOPENMX_CPU_MPI=OPENMPI \
  -DOPENMX_CPU_MPI_ROOT=/usr/lib64/openmpi \
  -DOPENMX_CPU_LINALG=OPENBLAS \
  -DCMAKE_C_COMPILER=/usr/bin/gcc14-gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/gcc14-g++ \
  -DCMAKE_Fortran_COMPILER=/usr/bin/gcc14-gfortran \
  -DCMAKE_INSTALL_PREFIX=/home/ec2-user/.local/opt/openmx-cpu \
  -DOPENMX_COPY_TO_WORK=OFF \
  -DOPENMX_BUILD_JOBS=8
cmake --build build/cpu --parallel 12
cmake --install build/cpu
