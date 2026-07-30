#!/bin/sh
# ============================================================================
# Build the gpusolver2 stack (SLATE with its bundled BLAS++/LAPACK++) from the
# release archive in third_party/dist.  Everything is built offline, with
# static libraries, and installed under $GPUSOLVER2_PREFIX.
#
# Driven by "make gpusolver2-stack" from source/Makefile, which passes every
# GPUSOLVER2_* variable below; standalone invocation works with the same
# environment variables.
#
# Toolchain notes:
#  - SLATE is compiled with the GNU host compilers (nvc++ 26.3 dies with an
#    internal error on several SLATE sources).  Its OpenMP references
#    (GOMP_*) are resolved by the NVHPC OpenMP runtime at the final openmx
#    link, so only one OpenMP runtime lives in the executable; the
#    slate_bridge pins the thread count during SLATE calls because that
#    runtime serializes nested parallel regions (see slate_bridge.cc).
#  - The CUDA::cublas/cusolver import targets are pre-seeded when the CUDA
#    math libraries live outside the CUDA toolkit tree (the NVHPC SDK
#    layout), where CMake's FindCUDAToolkit cannot discover them.
# ============================================================================
set -eu

log()  { printf '=== %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# ---- parameters ------------------------------------------------------------
GPUSOLVER2_DIST=${GPUSOLVER2_DIST:-$SCRIPT_DIR/dist}
GPUSOLVER2_BUILD=${GPUSOLVER2_BUILD:-$SCRIPT_DIR/../build/gpusolver2}
GPUSOLVER2_PREFIX=${GPUSOLVER2_PREFIX:-$SCRIPT_DIR/gpusolver2-install}
GPUSOLVER2_SLATE_VER=${GPUSOLVER2_SLATE_VER:-2025.05.28}
GPUSOLVER2_HOST_CC=${GPUSOLVER2_HOST_CC:-gcc}
GPUSOLVER2_HOST_CXX=${GPUSOLVER2_HOST_CXX:-g++}
GPUSOLVER2_HOST_FC=${GPUSOLVER2_HOST_FC:-gfortran}
GPUSOLVER2_GPU_ARCH=${GPUSOLVER2_GPU_ARCH:-}
GPUSOLVER2_MPI_BIN=${GPUSOLVER2_MPI_BIN:-}
GPUSOLVER2_MPI_LIBDIR=${GPUSOLVER2_MPI_LIBDIR:-}
GPUSOLVER2_CUDA_HOME=${GPUSOLVER2_CUDA_HOME:-}
GPUSOLVER2_NVCC=${GPUSOLVER2_NVCC:-$GPUSOLVER2_CUDA_HOME/bin/nvcc}
GPUSOLVER2_MATH_DIR=${GPUSOLVER2_MATH_DIR:-}
GPUSOLVER2_COMPILER_LIB=${GPUSOLVER2_COMPILER_LIB:-}
GPUSOLVER2_BLAS_LIBRARIES=${GPUSOLVER2_BLAS_LIBRARIES:-$GPUSOLVER2_COMPILER_LIB/libblas_lp64.so}
GPUSOLVER2_LAPACK_LIBRARIES=${GPUSOLVER2_LAPACK_LIBRARIES:-$GPUSOLVER2_COMPILER_LIB/liblapack_lp64.so;$GPUSOLVER2_COMPILER_LIB/libblas_lp64.so}
GPUSOLVER2_CMAKE=${GPUSOLVER2_CMAKE:-cmake}
GPUSOLVER2_JOBS=${GPUSOLVER2_JOBS:-$(nproc 2>/dev/null || echo 8)}

STAMP_DIR=$GPUSOLVER2_BUILD/stamp
SLATE_SRC=$GPUSOLVER2_BUILD/slate-$GPUSOLVER2_SLATE_VER
SLATE_BUILD=$GPUSOLVER2_BUILD/slate-build
SLATE_TARBALL=$GPUSOLVER2_DIST/slate-$GPUSOLVER2_SLATE_VER.tar.gz

[ -f "$SLATE_TARBALL" ] || die "bundled archive not found: $SLATE_TARBALL"
[ -n "$GPUSOLVER2_MPI_BIN" ]  || die "GPUSOLVER2_MPI_BIN is not set"
[ -n "$GPUSOLVER2_CUDA_HOME" ] || die "GPUSOLVER2_CUDA_HOME is not set"
[ -n "$GPUSOLVER2_GPU_ARCH" ] || die "GPUSOLVER2_GPU_ARCH is not set (no GPU detected?); e.g. use GPUSOLVER2_GPU_ARCH=90 for H100"

# ---- environment hygiene ---------------------------------------------------
# ~/.bashrc-style environments (Intel oneAPI in particular) poison CMake's
# compiler and MPI detection; start from a minimal, explicit environment.
unset I_MPI_ROOT ONEAPI_ROOT SETVARS_COMPLETED PKG_CONFIG_PATH CMAKE_PREFIX_PATH \
      CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LDFLAGS CFLAGS CXXFLAGS \
      CPPFLAGS FCFLAGS FFLAGS LIBS CC CXX FC F77 CUDA_HOME MKLROOT TBBROOT \
      CMPLR_ROOT NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS 2>/dev/null || true
PATH=$GPUSOLVER2_CUDA_HOME/bin:$GPUSOLVER2_MPI_BIN:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export PATH
LD_LIBRARY_PATH=$GPUSOLVER2_MPI_LIBDIR:$GPUSOLVER2_COMPILER_LIB:$GPUSOLVER2_CUDA_HOME/targets/x86_64-linux/lib:$GPUSOLVER2_MATH_DIR/lib
export LD_LIBRARY_PATH
OMPI_CC=$GPUSOLVER2_HOST_CC;  export OMPI_CC
OMPI_CXX=$GPUSOLVER2_HOST_CXX; export OMPI_CXX

command -v "$GPUSOLVER2_CMAKE" >/dev/null 2>&1 || die "cmake not found (SLATE needs cmake >= 3.18)"

mkdir -p "$GPUSOLVER2_BUILD" "$STAMP_DIR" "$GPUSOLVER2_PREFIX"

# ---- step 01: SLATE (+ bundled BLAS++/LAPACK++) ----------------------------
if [ -f "$STAMP_DIR/01_slate.done" ]; then
  log "SLATE already built (stamp found)"
else
  log "extracting $SLATE_TARBALL"
  rm -rf "$SLATE_SRC" "$SLATE_BUILD"
  tar -xzf "$SLATE_TARBALL" -C "$GPUSOLVER2_BUILD"
  [ -d "$SLATE_SRC" ] || die "unexpected tarball layout (no $SLATE_SRC)"

  # FindCUDAToolkit cannot see cuBLAS/cuSOLVER when they live in a separate
  # math-library tree (NVHPC SDK); pre-seed the library cache entries then.
  CUDA_PRESEED=""
  if [ -n "$GPUSOLVER2_MATH_DIR" ] && [ -f "$GPUSOLVER2_MATH_DIR/lib/libcublas.so" ] \
     && [ ! -e "$GPUSOLVER2_CUDA_HOME/targets/x86_64-linux/lib/libcublas.so" ]; then
    CUDA_PRESEED="-DCUDA_cublas_LIBRARY=$GPUSOLVER2_MATH_DIR/lib/libcublas.so \
      -DCUDA_cublasLt_LIBRARY=$GPUSOLVER2_MATH_DIR/lib/libcublasLt.so \
      -DCUDA_cusolver_LIBRARY=$GPUSOLVER2_MATH_DIR/lib/libcusolver.so \
      -DCUDA_cusparse_LIBRARY=$GPUSOLVER2_MATH_DIR/lib/libcusparse.so"
    MATH_INCLUDE="-I$GPUSOLVER2_MATH_DIR/include"
  else
    MATH_INCLUDE=""
  fi

  log "configuring SLATE (arch sm_$GPUSOLVER2_GPU_ARCH)"
  # shellcheck disable=SC2086
  "$GPUSOLVER2_CMAKE" -S "$SLATE_SRC" -B "$SLATE_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$GPUSOLVER2_PREFIX" \
    -DBUILD_SHARED_LIBS=OFF \
    -Dbuild_tests=OFF \
    -Dgpu_backend=cuda \
    -DCMAKE_CUDA_ARCHITECTURES="$GPUSOLVER2_GPU_ARCH" \
    -DCMAKE_C_COMPILER="$GPUSOLVER2_HOST_CC" \
    -DCMAKE_CXX_COMPILER="$GPUSOLVER2_HOST_CXX" \
    -DCMAKE_Fortran_COMPILER="$GPUSOLVER2_HOST_FC" \
    -DCMAKE_CUDA_COMPILER="$GPUSOLVER2_NVCC" \
    -DCMAKE_CUDA_HOST_COMPILER="$GPUSOLVER2_HOST_CXX" \
    -DMPI_C_COMPILER="$GPUSOLVER2_MPI_BIN/mpicc" \
    -DMPI_CXX_COMPILER="$GPUSOLVER2_MPI_BIN/mpicxx" \
    -DMPI_CXX_SKIP_MPICXX=ON \
    -DBLAS_LIBRARIES="$GPUSOLVER2_BLAS_LIBRARIES" \
    -DLAPACK_LIBRARIES="$GPUSOLVER2_LAPACK_LIBRARIES" \
    -DCMAKE_CXX_FLAGS="$MATH_INCLUDE" \
    -DCMAKE_CUDA_FLAGS="$MATH_INCLUDE" \
    $CUDA_PRESEED

  log "building SLATE"
  "$GPUSOLVER2_CMAKE" --build "$SLATE_BUILD" -j "$GPUSOLVER2_JOBS"
  log "installing SLATE into $GPUSOLVER2_PREFIX"
  "$GPUSOLVER2_CMAKE" --install "$SLATE_BUILD"

  [ -f "$GPUSOLVER2_PREFIX/lib/libslate.a" ] || die "libslate.a missing after install"
  touch "$STAMP_DIR/01_slate.done"
fi

log "gpusolver2 stack is ready under $GPUSOLVER2_PREFIX"
