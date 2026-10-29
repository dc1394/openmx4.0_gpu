#!/bin/sh
# Build the shared CPU/GPU ELPA and gpusolver2 dependency stack from the
# release archives bundled in third_party/dist:
#
#   * ELPA (CPU and NVIDIA GPU kernels) — distributed eigensolver
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
#                        (its directory need not be the MPI library directory)
#   GPUSOLVER2_ELPA_VER  ELPA version (tag new_release_<ver with underscores>)
#   GPUSOLVER2_CMAKE     cmake >= 3.24 for COSMA (bootstrapped from the
#                        bundled source archive when not available)
#   GPUSOLVER2_JOBS      parallel build jobs
#   GPUSOLVER2_TOOL_PATH directories searched before /usr/bin for the build
#                        tools, e.g. newer autotools than the system ones
#                        (see build_host_autotools.sh)
#   GPUSOLVER2_HOST_LIBDIR runtime library directory of GPUSOLVER2_HOST_CXX
#                        when it is not the system GCC
#
# ELPA is built from the GitLab tag archive, so autoconf (>= 2.71), automake,
# libtool, m4 and python3 must be available (standard on Ubuntu; module/apt or
# GPUSOLVER2_TOOL_PATH otherwise).
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
: "${GPUSOLVER2_ELPA_KERNELS:=auto}"
: "${GPUSOLVER2_CMAKE:=}"
: "${GPUSOLVER2_JOBS:=$(nproc 2>/dev/null || echo 8)}"
: "${GPUSOLVER2_TOOL_PATH:=}"
: "${GPUSOLVER2_HOST_LIBDIR:=}"

DIST=$GPUSOLVER2_DIST
SRC=$GPUSOLVER2_BUILD/src
BLD=$GPUSOLVER2_BUILD/build
STAMP=$GPUSOLVER2_BUILD/stamp
LOGS=$GPUSOLVER2_BUILD/logs
P=$GPUSOLVER2_PREFIX
J=$GPUSOLVER2_JOBS
mkdir -p "$SRC" "$BLD" "$STAMP" "$LOGS" "$P"

# Stamps from another compiler/CUDA/MPI installation cannot be reused.
# Keep extracted sources; rerun configure/CMake and compilation on a change.
printf '%s\n' "$GPUSOLVER2_COMPILER_LIB" "$GPUSOLVER2_MPI_BIN" "$GPUSOLVER2_MPI_LIBDIR" \
  "$GPUSOLVER2_CUDA_HOME" "$GPUSOLVER2_MATH_DIR" "$GPUSOLVER2_NVCC" \
  "$GPUSOLVER2_GPU_ARCH" "$GPUSOLVER2_ELPA_VER" "$GPUSOLVER2_HOST_CC" "$GPUSOLVER2_HOST_CXX" \
  "$GPUSOLVER2_PREFIX" "${GPUSOLVER2_COMPILER_BIN:-}" "$GPUSOLVER2_SCALAPACK_SO" \
  > "$STAMP/toolchain.tmp"
if ! cmp -s "$STAMP/toolchain.tmp" "$STAMP/toolchain.conf"; then
  rm -f "$STAMP/01_elpa.done" "$STAMP/02_cosma.done"
  # CMake/autoconf caches embed absolute compiler and MPI paths.
  rm -rf "$BLD/elpa" "$BLD/cosma"
  mv "$STAMP/toolchain.tmp" "$STAMP/toolchain.conf"
else
  rm -f "$STAMP/toolchain.tmp"
fi

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
NVCOMP_BIN=${GPUSOLVER2_COMPILER_BIN:-$(dirname "$GPUSOLVER2_COMPILER_LIB")/bin}
export NVHPC_CUDA_HOME="$GPUSOLVER2_CUDA_HOME"
export NVCOMPILER_CUDA_HOME="$GPUSOLVER2_CUDA_HOME"
export NVCOMPILER_MATH_LIBS_HOME="$GPUSOLVER2_MATH_DIR"
# Preserve the Makefile's NVCOMPILER_COMM_LIBS_HOME for the SDK MPI selector.
# The ScaLAPACK directory is searched after the MPI one: it may be another
# MPI installation's tree, whose libmpi must not shadow the selected MPI.
SCALAPACK_DIR=$(dirname "$GPUSOLVER2_SCALAPACK_SO")
export PATH="${GPUSOLVER2_TOOL_PATH:+$GPUSOLVER2_TOOL_PATH:}$GPUSOLVER2_MPI_BIN:$NVCOMP_BIN:$GPUSOLVER2_CUDA_HOME/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$GPUSOLVER2_MPI_LIBDIR:$SCALAPACK_DIR:$GPUSOLVER2_COMPILER_LIB:$GPUSOLVER2_CUDA_HOME/targets/x86_64-linux/lib:$GPUSOLVER2_MATH_DIR/lib${GPUSOLVER2_HOST_LIBDIR:+:$GPUSOLVER2_HOST_LIBDIR}"

fail() { echo "gpusolver2 stack: FAILED at step $1 (see $LOGS/$1.log)"; exit 1; }

run_step() {
  step=$1; shift
  [ -f "$STAMP/$step.done" ] && return 0
  echo "gpusolver2 stack: $step"
  # Do not put the subshell in an || condition: POSIX sh then disables
  # errexit throughout the called function and can stamp a failed build.
  ( set -e; "$@" ) > "$LOGS/$step.log" 2>&1
  step_status=$?
  [ "$step_status" -eq 0 ] || fail "$step"
  touch "$STAMP/$step.done"
}

# ---------- ELPA (autotools; needs autoreconf + python3)
step_elpa() {
  for tool in autoconf automake libtoolize m4 python3; do
    command -v $tool > /dev/null || { echo "ERROR: '$tool' is required to build the bundled ELPA"; exit 1; }
  done
  # configure survives between builds; its existence alone does not prove
  # that the extracted sources match a newly selected ELPA release. Accept
  # pre-stamp source trees through the release's own version metadata.
  elpa_source_version=""
  if [ -f "$SRC/elpa/.openmx-source-version" ]; then
    elpa_source_version=$(cat "$SRC/elpa/.openmx-source-version")
  elif [ -f "$SRC/elpa/elpa.spec" ]; then
    elpa_source_version=$(awk '$1 == "Version:" {print $2; exit}' "$SRC/elpa/elpa.spec")
  fi
  if [ "$elpa_source_version" != "$GPUSOLVER2_ELPA_VER" ] || [ ! -x "$SRC/elpa/configure" ]; then
    if [ ! -f "$DIST/elpa-$ELPA_TAG.zip" ]; then
      echo "ERROR: missing ELPA release archive: $DIST/elpa-$ELPA_TAG.zip"
      return 1
    fi
    rm -rf "$SRC/elpa" "$SRC"/elpa-"$ELPA_TAG"-*
    rm -rf "$BLD/elpa"
    ( cd "$SRC" && unzip -q "$DIST/elpa-$ELPA_TAG.zip" && mv elpa-"$ELPA_TAG"-* elpa ) || return 1
    ( cd "$SRC/elpa" && sh autogen.sh ) || return 1
  fi
  printf '%s\n' "$GPUSOLVER2_ELPA_VER" > "$SRC/elpa/.openmx-source-version" || return 1
  SM80=""
  [ "$GPUSOLVER2_GPU_ARCH" -ge 80 ] 2>/dev/null && SM80="--enable-nvidia-sm80-gpu"
  mkdir -p "$BLD/elpa" && cd "$BLD/elpa"
  # ELPA's nvcc rule does not see CPPFLAGS; the cuBLAS headers of an SDK
  # layout with separate math_libs have to reach it through NVCCFLAGS.
  # Generic real/complex CPU kernels support both elpa1 and elpa2.  The x86
  # SIMD kernels are C intrinsic templates that NVHPC's nvc cannot compile,
  # so with them enabled the C side goes through the MPI wrapper to the host
  # GCC (OMPI_CC) while the Fortran side keeps the SDK compiler; the kernel
  # set and the -m flags follow the build host's CPU like the application's
  # -march=native (the kernel block before run_step 01_elpa), and
  # openmx_elpa_cpu.c selects the kernel at run time with the generic
  # kernels as the fallback.
  case "$ELPA_KERNEL_CC" in
    gcc) export OMPI_CC="$GPUSOLVER2_HOST_CC" OMPI_CXX="$GPUSOLVER2_HOST_CXX" ;;
  esac
  "$SRC/elpa/configure" --prefix="$P" \
    FC="$GPUSOLVER2_MPI_BIN/mpif90" CC="$GPUSOLVER2_MPI_BIN/mpicc" CXX="$GPUSOLVER2_MPI_BIN/mpicxx" \
    FCFLAGS="-O2" CFLAGS="-O2 $ELPA_SIMD_CFLAGS" CXXFLAGS="-O2 $ELPA_SIMD_CFLAGS" \
    CPPFLAGS="-I$GPUSOLVER2_MATH_DIR/include -I$GPUSOLVER2_CUDA_HOME/targets/x86_64-linux/include" \
    NVCCFLAGS="-I$GPUSOLVER2_MATH_DIR/include" \
    LDFLAGS="-L$GPUSOLVER2_MPI_LIBDIR -L$SCALAPACK_DIR -L$GPUSOLVER2_COMPILER_LIB -L$GPUSOLVER2_MATH_DIR/lib" \
    LIBS="-lscalapack_lp64 -llapack_lp64 -lblas_lp64 -lstdc++" \
    --enable-nvidia-gpu-kernels --with-NVIDIA-GPU-compute-capability=sm_$GPUSOLVER2_GPU_ARCH $SM80 \
    --with-cuda-path="$GPUSOLVER2_CUDA_HOME" --disable-shared --enable-static \
    $ELPA_KERNEL_OPTS \
    --disable-c-tests --disable-cpp-tests --disable-fortran-tests --disable-Fortran-tests
  make -j "$J"
  make install
}
# The CPU kernels of ELPA's 2-stage solver (GPUSOLVER2_ELPA_KERNELS: auto =
# the SIMD kernels this build host can run, generic = portable kernels only).
# The MPI wrapper's own C compiler builds them when it can (GCC, Clang);
# NVHPC's nvc cannot, and the host GCC is used through OMPI_CC instead.
ELPA_KERNEL_OPTS="--enable-generic-kernels --disable-sse-assembly-kernels"
ELPA_SIMD_CFLAGS=""
ELPA_KERNEL_CC=wrapper
if [ "$GPUSOLVER2_ELPA_KERNELS" = auto ] && grep -qw avx2 /proc/cpuinfo 2>/dev/null \
   && grep -qw fma /proc/cpuinfo 2>/dev/null; then
  ELPA_KERNEL_OPTS="$ELPA_KERNEL_OPTS --enable-sse-kernels --enable-avx-kernels --enable-avx2-kernels"
  ELPA_SIMD_CFLAGS="-msse4.2 -mavx -mfma -mavx2"
  if grep -qw avx512f /proc/cpuinfo && grep -qw avx512dq /proc/cpuinfo \
     && grep -qw avx512bw /proc/cpuinfo && grep -qw avx512vl /proc/cpuinfo; then
    ELPA_KERNEL_OPTS="$ELPA_KERNEL_OPTS --enable-avx512-kernels"
    ELPA_SIMD_CFLAGS="$ELPA_SIMD_CFLAGS -mavx512f -mavx512cd -mavx512dq -mavx512bw -mavx512vl"
  else
    ELPA_KERNEL_OPTS="$ELPA_KERNEL_OPTS --disable-avx512-kernels"
  fi
  case "$("$GPUSOLVER2_MPI_BIN/mpicc" --showme:command 2>/dev/null)" in
    nvc*|pgcc*) ELPA_KERNEL_CC=gcc ;;
  esac
else
  ELPA_KERNEL_OPTS="$ELPA_KERNEL_OPTS --disable-sse-kernels --disable-avx-kernels --disable-avx2-kernels --disable-avx512-kernels"
fi
# The kernel set is part of ELPA's configuration: a change rebuilds ELPA
# alone (COSMA does not depend on it).
printf '%s\n' "$ELPA_KERNEL_OPTS" "$ELPA_SIMD_CFLAGS" "$ELPA_KERNEL_CC" > "$STAMP/elpa-kernels.tmp"
if ! cmp -s "$STAMP/elpa-kernels.tmp" "$STAMP/elpa-kernels.conf"; then
  rm -f "$STAMP/01_elpa.done"
  rm -rf "$BLD/elpa"
  mv "$STAMP/elpa-kernels.tmp" "$STAMP/elpa-kernels.conf"
else
  rm -f "$STAMP/elpa-kernels.tmp"
fi
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
  # The installation prefix may have changed since an earlier bootstrap,
  # or its installed binary may have been removed independently of stamps.
  rm -f "$STAMP/00_cmake_bootstrap.done"
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

# The OpenMX Makefile links $P/lib/*.a; without the explicit libdir, CMake
# installs into lib64 on RHEL-family systems.
COMMON="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DCMAKE_PREFIX_PATH=$P \
 -DCMAKE_INSTALL_LIBDIR=lib \
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
  # OpenMX's product hook in Tiled-MM (see the patch header): its double and
  # complex-double tile products call the application's GEMMul8 bridge.
  # Forward/reverse dry-runs reject an unexpected source tree and make
  # repeat builds idempotent; the hashes pin the one file the patch changes.
  tiledmm_file="$SRC/Tiled-MM/src/Tiled-MM/tiled_mm.cpp"
  if ! printf '%s  %s\n' "$TILEDMM_ORIGINAL" "$tiledmm_file" | sha256sum -c - >/dev/null 2>&1 &&
     ! printf '%s  %s\n' "$TILEDMM_PATCHED" "$tiledmm_file" | sha256sum -c - >/dev/null 2>&1; then
    echo "ERROR: unexpected Tiled-MM source hash; refusing to patch $tiledmm_file"
    return 1
  fi
  if patch --batch --fuzz=0 --forward --dry-run -d "$SRC/Tiled-MM" -p1 < "$TILEDMM_PATCH" >/dev/null 2>&1; then
    patch --batch --fuzz=0 --forward -d "$SRC/Tiled-MM" -p1 < "$TILEDMM_PATCH"
  elif ! patch --batch --fuzz=0 --reverse --dry-run -d "$SRC/Tiled-MM" -p1 < "$TILEDMM_PATCH" >/dev/null 2>&1; then
    echo "ERROR: Tiled-MM source does not match $TILEDMM_PATCH"
    return 1
  fi
  printf '%s  %s\n' "$TILEDMM_PATCHED" "$tiledmm_file" | sha256sum -c -
  # COSMA_SCALAPACK=CUSTOM forces the host-BLAS vendor to "auto", whose
  # OPENBLAS probe "succeeds" with NOTFOUND libraries on machines without
  # OpenBLAS (FindOPENBLAS.cmake creates its target unconditionally).
  # Predefining the probe's cache variables pins it to the NVHPC LP64 BLAS
  # instead -- the same family the final openmx link uses.
  GenericBLAS_ROOT=$GPUSOLVER2_COMPILER_LIB \
  $CMAKE -S "$SRC/cosma" -B "$BLD/cosma" $COMMON $CUDAFLAGS $MPIARGS \
    -DCOSMA_BLAS=CUDA -DCOSMA_SCALAPACK=CUSTOM \
    -DCOSMA_SCALAPACK_LINK_LIBRARIES="$GPUSOLVER2_SCALAPACK_SO" \
    -DCOSMA_OPENBLAS_LINK_LIBRARIES="$GPUSOLVER2_COMPILER_LIB/libblas_lp64.so" \
    -DCOSMA_OPENBLAS_INCLUDE_DIRS="${GPUSOLVER2_COMPILER_LIB%/lib}/include" \
    -DCOSMA_WITH_TESTS=OFF -DCOSMA_WITH_APPS=OFF -DCOSMA_WITH_BENCHMARKS=OFF \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_COSTA="$SRC/COSTA" \
    "-DFETCHCONTENT_SOURCE_DIR_TILED-MM=$SRC/Tiled-MM"
  $CMAKE --build "$BLD/cosma" -j "$J"
  $CMAKE --install "$BLD/cosma"
}
# The Tiled-MM patch is part of COSMA's configuration: a change rebuilds
# COSMA alone (ELPA does not depend on it).
TILEDMM_PATCH=$(dirname "$DIST")/tiledmm-openmx-gemm-hook.patch
TILEDMM_ORIGINAL=9b33855deba33b276bd1d95db921fba95a81152018f477255d02c86ab3b3e5fb
TILEDMM_PATCHED=908237509e8d66cfdf73da7acc6501aa35adf62dd6a9126749c1e8071d9d473c
[ -f "$TILEDMM_PATCH" ] || { echo "ERROR: missing $TILEDMM_PATCH"; exit 1; }
sha256sum < "$TILEDMM_PATCH" > "$STAMP/cosma-patches.tmp"
if ! cmp -s "$STAMP/cosma-patches.tmp" "$STAMP/cosma-patches.conf"; then
  rm -f "$STAMP/02_cosma.done"
  rm -rf "$BLD/cosma"
  mv "$STAMP/cosma-patches.tmp" "$STAMP/cosma-patches.conf"
else
  rm -f "$STAMP/cosma-patches.tmp"
fi
run_step 02_cosma step_cosma

echo "gpusolver2 stack: all components installed into $P"
