#!/usr/bin/env bash
set -euo pipefail
source /home/ec2-user/openmx-build/scripts/environment.sh
root=/home/ec2-user/openmx-build
blas=/home/ec2-user/.local/opt/openblas-0.3.34-sve256/lib
arch_flags='-mcpu=neoverse-v1 -msve-vector-bits=scalable -fno-fast-math -ffp-contract=off'
test -f "$blas/libopenblas.so"
cmake -S "$root/deps/scalapack-2.2.3" -B "$root/scalapack" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc14-gcc -DCMAKE_Fortran_COMPILER=/usr/bin/gcc14-gfortran \
  -DMPI_C_COMPILER=/usr/lib64/openmpi/bin/mpicc -DMPI_Fortran_COMPILER=/usr/lib64/openmpi/bin/mpifort \
  -DCMAKE_C_FLAGS="$arch_flags" -DCMAKE_Fortran_FLAGS="$arch_flags" \
  -DBUILD_SHARED_LIBS=OFF -DSCALAPACK_BUILD_TESTS=OFF \
  -DBLAS_LIBRARIES="$blas/libopenblas.so" -DLAPACK_LIBRARIES="$blas/libopenblas.so" \
  > "$root/logs/scalapack-configure.log" 2>&1
cmake --build "$root/scalapack" --target scalapack --parallel 12 > "$root/logs/scalapack-build.log" 2>&1
printf 'ScaLAPACK build complete\n'
elpa_src="$root/deps/elpa-new_release_2026_02_002-202e14d535cb07e1193fdc5e5cea726eb5faa44f"
export CC=mpicc CXX=mpicxx FC=mpifort
export CFLAGS="$arch_flags -O2 -fPIC -march=armv8-a+sve -msve-vector-bits=256"
export CXXFLAGS="$CFLAGS"
export FCFLAGS='-O2 -fPIC'
export LIBS="-L$root/scalapack/lib -lscalapack -L$blas -lopenblas -lm"
export LD_LIBRARY_PATH="$blas:/usr/lib64/openmpi/lib"
export LDFLAGS="" CPPFLAGS="" CPATH="" C_INCLUDE_PATH="" CPLUS_INCLUDE_PATH=""
cd "$elpa_src"
sh autogen.sh > "$root/logs/elpa-autogen.log" 2>&1
mkdir -p "$root/elpa"
cd "$root/elpa"
"$elpa_src/configure" --prefix=/home/ec2-user/.local/opt/elpa-2026.02.002-sve256 \
  --enable-openmp --disable-shared --enable-static --with-pic \
  --disable-nvidia-gpu-kernels --disable-amd-gpu-kernels --disable-intel-gpu-sycl-kernels \
  --enable-generic-kernels --disable-sse-assembly-kernels \
  --disable-sse-kernels --disable-avx-kernels --disable-avx2-kernels --disable-avx512-kernels \
  --enable-neon-arch64-kernels --enable-sve256-kernels \
  --with-test-programs=no --disable-c-tests --disable-cpp-tests \
  --disable-fortran-tests --disable-Fortran-tests \
  > "$root/logs/elpa-configure.log" 2>&1
make -j12 libelpa_openmp.la > "$root/logs/elpa-build.log" 2>&1
make -j12 >> "$root/logs/elpa-build.log" 2>&1
make install > "$root/logs/elpa-install.log" 2>&1
printf 'ELPA build and installation complete\n'
