#!/usr/bin/env bash
# Match the compiler used to build the CPU application in test subprocesses.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
test_build=${OPENMX_LOCAL_TEST_BUILD:-build/cpu}
mkdir -p "$test_build/test-toolchain/bin"
for compiler in cc gcc; do
  ln -sfn /usr/bin/gcc14-gcc "$test_build/test-toolchain/bin/$compiler"
done
for compiler in c++ g++; do
  ln -sfn /usr/bin/gcc14-g++ "$test_build/test-toolchain/bin/$compiler"
done
ln -sfn /usr/bin/gcc14-gfortran "$test_build/test-toolchain/bin/gfortran"
export PATH="$PWD/$test_build/test-toolchain/bin:/usr/lib64/openmpi/bin:/home/ec2-user/.local/opt/openmx-autotools/bin:/home/ec2-user/.local/bin:$PATH"
export CC=/usr/bin/gcc14-gcc CXX=/usr/bin/gcc14-g++ FC=/usr/bin/gcc14-gfortran
export OMPI_CC="$CC" OMPI_CXX="$CXX" OMPI_FC="$FC"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ctest --test-dir "$test_build" --output-on-failure --parallel 8 "$@"
