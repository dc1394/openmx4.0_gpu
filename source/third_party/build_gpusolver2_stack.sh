#!/bin/sh
# Build the scf.eigen.lib=gpusolver2 dependency stack from the release
# archives bundled in third_party/dist:
#
#   * DLA-Future (+ pika, Umpire, whip, blaspp, lapackpp, fmt,
#     Boost.Context)                 — distributed GPU eigensolver
#   * COSMA (+ COSTA, Tiled-MM)      — distributed pdgemm/pzgemm on the GPU
#
# Everything is driven by the OpenMX Makefile ("make gpusolver2-stack"); the
# knobs below can be overridden from the make command line / environment.
#
#   GPUSOLVER2_DIST      directory holding the bundled archives
#   GPUSOLVER2_BUILD     scratch build directory (safe to delete)
#   GPUSOLVER2_PREFIX    installation prefix that openmx links against
#   GPUSOLVER2_HOST_CC / GPUSOLVER2_HOST_CXX
#                        host GCC used for the C++ stack
#   GPUSOLVER2_NVCC      CUDA compiler
#   GPUSOLVER2_GPU_ARCH  GPU compute capability (90 for H100, 120 for RTX 5080)
#   GPUSOLVER2_MPI_BIN / GPUSOLVER2_MPI_LIBDIR
#                        the SAME MPI openmx is built with (HPC-X wrappers);
#                        hwloc headers/library are also taken from this
#                        Open MPI installation
#   GPUSOLVER2_CUDA_HOME / GPUSOLVER2_MATH_DIR / GPUSOLVER2_COMPILER_LIB
#                        NVHPC CUDA, math_libs (cublas), compiler lib dirs
#   GPUSOLVER2_SCALAPACK_SO shared ScaLAPACK used for CMake/configure probes
#                        and for the DLA-Future ScaLAPACK-like C API
#   GPUSOLVER2_CMAKE     cmake >= 3.24 (bootstrapped from the bundled source
#                        archive when not available)
#   GPUSOLVER2_JOBS      parallel build jobs
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
V_FMT=11.1.4
V_SPDLOG=1.15.3
V_BOOST=1.87.0
V_PIKA=0.31.0
V_UMPIRE=2025.12.0
V_WHIP=0.3.0
V_BLASPP=2025.05.28
V_LAPACKPP=2025.05.28
V_DLAF=0.10.0
SHA_COSTA=2484769535772f807d402901ffca63bb6678dd42
SHA_TILEDMM=0eb75179e670a04c649b50ae5e91bb71b43e4d06

# hwloc (needed by pika) comes from the same Open MPI installation openmx
# links against, so build and run time agree on one hwloc.
HWLOC_DIR=$(dirname "$GPUSOLVER2_MPI_LIBDIR")

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

# ---------- cmake >= 3.24: use $GPUSOLVER2_CMAKE, else system cmake, else bootstrap
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
 -DCMAKE_CUDA_HOST_COMPILER=$GPUSOLVER2_HOST_CXX -DCUDAToolkit_ROOT=$GPUSOLVER2_CUDA_HOME"
MPIARGS="-DMPI_C_COMPILER=$GPUSOLVER2_MPI_BIN/mpicc -DMPI_CXX_COMPILER=$GPUSOLVER2_MPI_BIN/mpicxx \
 -DMPI_CXX_SKIP_MPICXX=ON"

# the CMake builds go through the MPI wrappers with GCC
export OMPI_CC=$GPUSOLVER2_HOST_CC
export OMPI_CXX=$GPUSOLVER2_HOST_CXX

# Unpack an archive into $SRC/<name> and drop the matching $BLD/<name>,
# so a version bump never configures against a stale CMake cache.
unpack_tar() { # unpack_tar <archive> <name>
  rm -rf "$SRC/$2" "$BLD/$2" && mkdir -p "$SRC/$2"
  tar xf "$1" -C "$SRC/$2" --strip-components=1
}

# ---------- fmt (required by pika)
step_fmt() {
  unpack_tar "$DIST/fmt-$V_FMT.tar.gz" "fmt"
  $CMAKE -S "$SRC/fmt" -B "$BLD/fmt" $COMMON \
    -DFMT_DOC=OFF -DFMT_TEST=OFF
  $CMAKE --build "$BLD/fmt" -j "$J"
  $CMAKE --install "$BLD/fmt"
}
run_step 01_fmt step_fmt

# ---------- spdlog (logging library required by pika; uses the fmt above)
step_spdlog() {
  unpack_tar "$DIST/spdlog-$V_SPDLOG.tar.gz" "spdlog"
  $CMAKE -S "$SRC/spdlog" -B "$BLD/spdlog" $COMMON \
    -DSPDLOG_FMT_EXTERNAL=ON -DSPDLOG_BUILD_EXAMPLE=OFF -DSPDLOG_BUILD_TESTS=OFF
  $CMAKE --build "$BLD/spdlog" -j "$J"
  $CMAKE --install "$BLD/spdlog"
}
run_step 02_spdlog step_spdlog

# ---------- Boost headers + Boost.Context (required by pika)
step_boost() {
  unpack_tar "$DIST/boost-$V_BOOST-b2-nodocs.tar.xz" "boost"
  cd "$SRC/boost"
  ./bootstrap.sh --prefix="$P"
  ./b2 --with-context variant=release link=static threading=multi \
       cxxflags=-fPIC -j "$J" --prefix="$P" install
}
run_step 03_boost step_boost

# ---------- whip (CUDA/HIP abstraction used by pika and DLA-Future;
#            header-only)
step_whip() {
  unpack_tar "$DIST/whip-$V_WHIP.tar.gz" "whip"
  $CMAKE -S "$SRC/whip" -B "$BLD/whip" $COMMON $CUDAFLAGS \
    -DWHIP_BACKEND=CUDA
  $CMAKE --build "$BLD/whip" -j "$J"
  $CMAKE --install "$BLD/whip"
}
run_step 04_whip step_whip

# ---------- pika (task runtime of DLA-Future; version pinned by the
#            DLA-Future 0.10.0 CI, built with the system allocator)
step_pika() {
  unpack_tar "$DIST/pika-$V_PIKA.tar.gz" "pika"
  $CMAKE -S "$SRC/pika" -B "$BLD/pika" $COMMON $CUDAFLAGS $MPIARGS \
    -DPIKA_WITH_CUDA=ON -DPIKA_WITH_MPI=ON -DPIKA_WITH_MALLOC=system \
    -DPIKA_WITH_EXAMPLES=OFF -DPIKA_WITH_TESTS=OFF \
    -DHWLOC_ROOT="$HWLOC_DIR"
  $CMAKE --build "$BLD/pika" -j "$J"
  $CMAKE --install "$BLD/pika"
}
run_step 05_pika step_pika

# ---------- Umpire (memory pools of DLA-Future; camp/BLT are bundled in
#            the release tarball; fmt_DIR points at the fmt built above,
#            otherwise Umpire installs its own bundled fmt over it)
step_umpire() {
  unpack_tar "$DIST/umpire-$V_UMPIRE.tar.gz" "umpire"
  $CMAKE -S "$SRC/umpire" -B "$BLD/umpire" $COMMON $CUDAFLAGS \
    -DENABLE_CUDA=ON -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF \
    -DENABLE_BENCHMARKS=OFF -DUMPIRE_ENABLE_C=OFF -DUMPIRE_ENABLE_FORTRAN=OFF \
    -DUMPIRE_ENABLE_TOOLS=OFF -Dfmt_DIR="$P/lib/cmake/fmt"
  $CMAKE --build "$BLD/umpire" -j "$J"
  $CMAKE --install "$BLD/umpire"
}
run_step 06_umpire step_umpire

# ---------- blaspp / lapackpp (host BLAS/LAPACK C++ wrappers of DLA-Future;
#            the GPU work goes through cuBLAS/cuSOLVER directly, so the GPU
#            backend of blaspp stays off)
step_blaspp() {
  unpack_tar "$DIST/blaspp-$V_BLASPP.tar.gz" "blaspp"
  $CMAKE -S "$SRC/blaspp" -B "$BLD/blaspp" $COMMON \
    -Dgpu_backend=none -Dbuild_tests=OFF \
    -DBLAS_LIBRARIES="$GPUSOLVER2_COMPILER_LIB/libblas_lp64.so"
  $CMAKE --build "$BLD/blaspp" -j "$J"
  $CMAKE --install "$BLD/blaspp"
}
run_step 07_blaspp step_blaspp

step_lapackpp() {
  unpack_tar "$DIST/lapackpp-$V_LAPACKPP.tar.gz" "lapackpp"
  $CMAKE -S "$SRC/lapackpp" -B "$BLD/lapackpp" $COMMON \
    -Dgpu_backend=none -Dbuild_tests=OFF \
    -DLAPACK_LIBRARIES="$GPUSOLVER2_COMPILER_LIB/liblapack_lp64.so;$GPUSOLVER2_COMPILER_LIB/libblas_lp64.so"
  $CMAKE --build "$BLD/lapackpp" -j "$J"
  $CMAKE --install "$BLD/lapackpp"
}
run_step 08_lapackpp step_lapackpp

# ---------- DLA-Future (distributed GPU eigensolver, ScaLAPACK-like C API)
step_dlaf() {
  rm -rf "$SRC/dlaf" "$SRC"/DLA-Future-* "$BLD/dlaf"
  ( cd "$SRC" && unzip -q "$DIST/DLA-Future-$V_DLAF.zip" && mv "DLA-Future-$V_DLAF" dlaf )
  $CMAKE -S "$SRC/dlaf" -B "$BLD/dlaf" $COMMON $CUDAFLAGS $MPIARGS \
    -DDLAF_WITH_CUDA=ON -DDLAF_WITH_SCALAPACK=ON \
    -DDLAF_SCALAPACK_LIBRARY="$GPUSOLVER2_SCALAPACK_SO" \
    -DDLAF_LAPACK_LIBRARY="$GPUSOLVER2_COMPILER_LIB/liblapack_lp64.so;$GPUSOLVER2_COMPILER_LIB/libblas_lp64.so" \
    -DDLAF_BUILD_MINIAPPS=OFF -DDLAF_BUILD_TESTING=OFF -DDLAF_BUILD_DOC=OFF \
    -DHWLOC_ROOT="$HWLOC_DIR"
  $CMAKE --build "$BLD/dlaf" -j "$J"
  $CMAKE --install "$BLD/dlaf"
}
run_step 09_dlaf step_dlaf

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
run_step 10_cosma step_cosma

echo "gpusolver2 stack: all components installed into $P"
