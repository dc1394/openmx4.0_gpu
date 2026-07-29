#!/bin/sh
# Build the scf.eigen.lib=gpusolver2 dependency stack from the release
# archives bundled in third_party/dist:
#
#   * ELPA (with NVIDIA GPU kernels)  — distributed eigensolver
#   * COSMA (+ COSTA, Tiled-MM)       — distributed pdgemm/pzgemm on the GPU
#
# Everything is driven by the OpenMX Makefile ("make gpusolver2-stack"); the
# knobs below can be overridden from the make command line / environment.
#
#   GPUSOLVER2_DIST      directory holding the bundled archives
#   GPUSOLVER2_BUILD     scratch build directory (safe to delete)
#   GPUSOLVER2_PREFIX    installation prefix that openmx links against
#   GPUSOLVER2_HOST_CC / GPUSOLVER2_HOST_CXX
#                        host GCC used for COSMA (C++)
#   GPUSOLVER2_NVCC      CUDA compiler
#   GPUSOLVER2_GPU_ARCH  GPU compute capability (90 for H100, 120 for RTX 5080)
#   GPUSOLVER2_MPI_BIN / GPUSOLVER2_MPI_LIBDIR
#                        the SAME MPI openmx is built with (HPC-X wrappers)
#   GPUSOLVER2_CUDA_HOME / GPUSOLVER2_MATH_DIR / GPUSOLVER2_COMPILER_LIB
#                        NVHPC CUDA, math_libs (cublas), compiler lib dirs
#   GPUSOLVER2_SCALAPACK_SO shared ScaLAPACK used for CMake/configure probes
#   GPUSOLVER2_ELPA_VER  ELPA version (tag new_release_<ver with underscores>)
#   GPUSOLVER2_CMAKE     cmake >= 3.24 for COSMA (bootstrapped from the
#                        bundled source archive when not available)
#   GPUSOLVER2_JOBS      parallel build jobs
#
# ELPA is built from the GitLab tag archive, so autoconf/automake/libtool/m4
# and python3 must be available (standard on Ubuntu; module/apt otherwise).
#
# Each step leaves a stamp in $GPUSOLVER2_BUILD/stamp and is skipped when the
# stamp exists, so a failed build resumes where it stopped.
set -u

: "${GPUSOLVER2_DIST:?set by the Makefile}"
: "${GPUSOLVER2_BUILD:?set by the Makefile}"
: "${GPUSOLVER2_PREFIX:?set by the Makefile}"
: "${GPUSOLVER2_HOST_CC:=gcc}"
: "${GPUSOLVER2_HOST_CXX:=g++}"
: "${GPUSOLVER2_NVCC:?set by the Makefile}"
: "${GPUSOLVER2_GPU_ARCH:?set by the Makefile}"
: "${GPUSOLVER2_MPI_BIN:?set by the Makefile}"
: "${GPUSOLVER2_MPI_LIBDIR:?set by the Makefile}"
: "${GPUSOLVER2_CUDA_HOME:?set by the Makefile}"
: "${GPUSOLVER2_MATH_DIR:?set by the Makefile}"
: "${GPUSOLVER2_COMPILER_LIB:?set by the Makefile}"
: "${GPUSOLVER2_SCALAPACK_SO:?set by the Makefile}"
: "${GPUSOLVER2_ELPA_VER:=2026.02.002}"
: "${GPUSOLVER2_CMAKE:=}"
: "${GPUSOLVER2_JOBS:=$(nproc 2>/dev/null || echo 8)}"

DIST=$GPUSOLVER2_DIST
SRC=$GPUSOLVER2_BUILD/src
BLD=$GPUSOLVER2_BUILD/build
STAMP=$GPUSOLVER2_BUILD/stamp
LOGS=$GPUSOLVER2_BUILD/logs
P=$GPUSOLVER2_PREFIX
J=$GPUSOLVER2_JOBS
mkdir -p "$SRC" "$BLD" "$STAMP" "$LOGS" "$P"

V_CMAKE=3.31.12
V_COSMA=2.8.4
SHA_COSTA=2484769535772f807d402901ffca63bb6678dd42
SHA_TILEDMM=0eb75179e670a04c649b50ae5e91bb71b43e4d06
ELPA_TAG=new_release_$(echo "$GPUSOLVER2_ELPA_VER" | tr . _)

# Scrub environment that is known to poison the builds (Intel oneAPI
# setvars.sh exports I_MPI_ROOT/PKG_CONFIG_PATH and hijacks FindMPI).
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FCFLAGS
unset I_MPI_ROOT MPI_HOME MPI_ROOT PKG_CONFIG_PATH ONEAPI_ROOT CMAKE_PREFIX_PATH
unset FI_PROVIDER_PATH CCL_ROOT TBBROOT MKLROOT NLSPATH CMPLR_ROOT
NVCOMP_BIN=$(dirname "$GPUSOLVER2_NVCC")/../../../compilers/bin
export PATH="$GPUSOLVER2_MPI_BIN:$NVCOMP_BIN:$GPUSOLVER2_CUDA_HOME/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$GPUSOLVER2_MPI_LIBDIR:$GPUSOLVER2_COMPILER_LIB:$GPUSOLVER2_CUDA_HOME/targets/x86_64-linux/lib:$GPUSOLVER2_MATH_DIR/lib"

fail() { echo "gpusolver2 stack: FAILED at step $1 (see $LOGS/$1.log)"; exit 1; }

run_step() {
  step=$1; shift
  [ -f "$STAMP/$step.done" ] && return 0
  echo "gpusolver2 stack: $step"
  ( "$@" ) > "$LOGS/$step.log" 2>&1 || fail "$step"
  touch "$STAMP/$step.done"
}

# ---------- ELPA (autotools; needs autoreconf + python3)
step_elpa() {
  for tool in autoconf automake libtoolize m4 python3; do
    command -v $tool > /dev/null || { echo "ERROR: '$tool' is required to build the bundled ELPA"; exit 1; }
  done
  if [ ! -x "$SRC/elpa/configure" ]; then
    rm -rf "$SRC/elpa" "$SRC"/elpa-"$ELPA_TAG"-*
    ( cd "$SRC" && unzip -q "$DIST/elpa-$ELPA_TAG.zip" && mv elpa-"$ELPA_TAG"-* elpa )
    ( cd "$SRC/elpa" && sh autogen.sh )
  fi
  SM80=""
  [ "$GPUSOLVER2_GPU_ARCH" -ge 80 ] 2>/dev/null && SM80="--enable-nvidia-sm80-gpu"
  mkdir -p "$BLD/elpa" && cd "$BLD/elpa"
  "$SRC/elpa/configure" --prefix="$P" \
    FC="$GPUSOLVER2_MPI_BIN/mpif90" CC="$GPUSOLVER2_MPI_BIN/mpicc" CXX="$GPUSOLVER2_MPI_BIN/mpicxx" \
    FCFLAGS="-O2" CFLAGS="-O2" CXXFLAGS="-O2" \
    CPPFLAGS="-I$GPUSOLVER2_MATH_DIR/include -I$GPUSOLVER2_CUDA_HOME/targets/x86_64-linux/include" \
    LDFLAGS="-L$GPUSOLVER2_MPI_LIBDIR -L$GPUSOLVER2_COMPILER_LIB -L$GPUSOLVER2_MATH_DIR/lib" \
    LIBS="-lscalapack_lp64 -llapack_lp64 -lblas_lp64 -lstdc++" \
    --enable-nvidia-gpu-kernels --with-NVIDIA-GPU-compute-capability=sm_$GPUSOLVER2_GPU_ARCH $SM80 \
    --with-cuda-path="$GPUSOLVER2_CUDA_HOME" --disable-shared --enable-static \
    --disable-sse-assembly --disable-sse --disable-avx --disable-avx2 --disable-avx512 \
    --disable-c-tests --disable-cpp-tests --disable-fortran-tests --disable-Fortran-tests
  make -j "$J"
  make install

  # The OpenMX source tree embeds ELPA 2018.05 for the elpa1/elpa2 keywords,
  # and its Fortran module symbols (elpa_utilities_*, elpa2_workload_*,
  # aligned_mem_) collide with libelpa.a.  Fuse the new library into one
  # relocatable object and localize everything except the C API used by
  # elpa_cosma_bridge.c (the mpi_fortran_* commons must stay global so that
  # they keep merging with the MPI Fortran runtime).
  cd "$P/lib"
  ld -r -o elpa_whole.o --whole-archive libelpa.a --no-whole-archive
  nm -g --defined-only elpa_whole.o | awk '{print $3}' | grep '^mpi_fortran_' | sort -u > elpa_keep.txt
  printf 'elpa_init\nelpa_uninit\nelpa_allocate\nelpa_deallocate\nelpa_setup\nelpa_set_integer\nelpa_set_double\nelpa_eigenvectors_double\nelpa_eigenvectors_double_complex\nelpa_strerr\n' >> elpa_keep.txt
  objcopy --keep-global-symbols=elpa_keep.txt elpa_whole.o elpa_isolated.o
  rm -f elpa_whole.o
}
run_step 01_elpa step_elpa

# ---------- cmake >= 3.24 for COSMA: use $GPUSOLVER2_CMAKE, else system cmake, else bootstrap
cmake_ok() {
  v=$("$1" --version 2>/dev/null | sed -n '1s/[^0-9]*\([0-9][0-9.]*\).*/\1/p')
  [ -n "$v" ] || return 1
  maj=${v%%.*}; rest=${v#*.}; min=${rest%%.*}
  [ "$maj" -gt 3 ] 2>/dev/null && return 0
  [ "$maj" -eq 3 ] 2>/dev/null && [ "$min" -ge 24 ] 2>/dev/null && return 0
  return 1
}

CMAKE=""
if [ -n "$GPUSOLVER2_CMAKE" ] && cmake_ok "$GPUSOLVER2_CMAKE"; then
  CMAKE=$GPUSOLVER2_CMAKE
elif cmake_ok cmake; then
  CMAKE=cmake
elif [ -x "$P/cmake-bootstrap/bin/cmake" ]; then
  CMAKE=$P/cmake-bootstrap/bin/cmake
else
  step_cmake_bootstrap() {
    rm -rf "$SRC/cmake" && mkdir -p "$SRC/cmake"
    tar xzf "$DIST/cmake-$V_CMAKE.tar.gz" -C "$SRC/cmake" --strip-components=1
    cd "$SRC/cmake"
    ./bootstrap --prefix="$P/cmake-bootstrap" --parallel="$J" -- -DCMAKE_BUILD_TYPE=Release
    make -j "$J"
    make install
  }
  run_step 00_cmake_bootstrap step_cmake_bootstrap
  CMAKE=$P/cmake-bootstrap/bin/cmake
fi
echo "gpusolver2 stack: using cmake: $CMAKE ($($CMAKE --version | head -n1))"

COMMON="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DCMAKE_PREFIX_PATH=$P \
 -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
 -DCMAKE_C_COMPILER=$GPUSOLVER2_HOST_CC -DCMAKE_CXX_COMPILER=$GPUSOLVER2_HOST_CXX \
 -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
CUDAFLAGS="-DCMAKE_CUDA_COMPILER=$GPUSOLVER2_NVCC -DCMAKE_CUDA_ARCHITECTURES=$GPUSOLVER2_GPU_ARCH \
 -DCMAKE_CUDA_HOST_COMPILER=$GPUSOLVER2_HOST_CXX"
MPIARGS="-DMPI_C_COMPILER=$GPUSOLVER2_MPI_BIN/mpicc -DMPI_CXX_COMPILER=$GPUSOLVER2_MPI_BIN/mpicxx \
 -DMPI_CXX_SKIP_MPICXX=ON"

# the COSMA CMake builds go through the MPI wrappers with GCC
export OMPI_CC=$GPUSOLVER2_HOST_CC
export OMPI_CXX=$GPUSOLVER2_HOST_CXX

# ---------- COSMA (communication-optimal pdgemm/pzgemm, GPU backend;
#            COSTA and Tiled-MM are provided offline at the pinned commits)
step_cosma() {
  [ -d "$SRC/cosma" ] || { cd "$SRC" && unzip -q "$DIST/cosma-v$V_COSMA.zip" && mv "COSMA-$V_COSMA" cosma; }
  [ -d "$SRC/COSTA" ] || { cd "$SRC" && unzip -q "$DIST/COSTA-$SHA_COSTA.zip" && mv "COSTA-$SHA_COSTA" COSTA; }
  [ -d "$SRC/Tiled-MM" ] || { cd "$SRC" && unzip -q "$DIST/Tiled-MM-$SHA_TILEDMM.zip" && mv "Tiled-MM-$SHA_TILEDMM" Tiled-MM; }
  GenericBLAS_ROOT=$GPUSOLVER2_COMPILER_LIB \
  $CMAKE -S "$SRC/cosma" -B "$BLD/cosma" $COMMON $CUDAFLAGS $MPIARGS \
    -DCOSMA_BLAS=CUDA -DCOSMA_SCALAPACK=CUSTOM \
    -DCOSMA_SCALAPACK_LINK_LIBRARIES="$GPUSOLVER2_SCALAPACK_SO" \
    -DCOSMA_WITH_TESTS=OFF -DCOSMA_WITH_APPS=OFF -DCOSMA_WITH_BENCHMARKS=OFF \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_COSTA="$SRC/COSTA" \
    "-DFETCHCONTENT_SOURCE_DIR_TILED-MM=$SRC/Tiled-MM"
  $CMAKE --build "$BLD/cosma" -j "$J"
  $CMAKE --install "$BLD/cosma"
}
run_step 02_cosma step_cosma

echo "gpusolver2 stack: all components installed into $P"
