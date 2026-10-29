#!/bin/sh
# Build the shared CPU/GPU ELPA and gpusolver2 dependency stack from the
# release archives bundled in third_party/dist:
#
#   * ELPA (CPU and AMD GPU kernels)    — distributed eigensolver
#   * COSMA (+ COSTA, Tiled-MM)       — distributed pdgemm/pzgemm on the GPU
#
# Everything is driven by the OpenMX Makefile ("make gpusolver2-stack"); the
# knobs below can be overridden from the make command line / environment.
#
#   GPUSOLVER2_HOST_CONFIG generated CMake MPI/math/runtime configuration
#   GPUSOLVER2_DIST      directory holding the bundled archives
#   GPUSOLVER2_BUILD     scratch build directory (safe to delete)
#   GPUSOLVER2_PREFIX    installation prefix that openmx links against
#   GPUSOLVER2_HOST_CC / GPUSOLVER2_HOST_CXX
#                        host compilers (CMake uses ROCm Clang)
#   GPUSOLVER2_HIPCC     HIP compiler (compiles the ELPA/Tiled-MM kernels)
#   GPUSOLVER2_GPU_ARCH  GPU architecture (gfx942 for MI300A, gfx1200 ...)
#   GPUSOLVER2_MPI_BIN / GPUSOLVER2_MPI_LIBDIR
#                        the SAME MPI openmx is built with; CMake uses mpif.h
#                        if mpi.mod does not match the Fortran compiler
#   GPUSOLVER2_ROCM      ROCm installation (hip, rocblas, amdhip64)
#   GPUSOLVER2_AOCL_LIB / GPUSOLVER2_AOCL_INC
#                        AOCL ScaLAPACK/libFLAME/BLIS used for the host BLAS
#   GPUSOLVER2_AOCC_LIB  AOCC runtime libs AOCL's ScaLAPACK was built against
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
: "${GPUSOLVER2_HIPCC:?set by the Makefile}"
: "${GPUSOLVER2_GPU_ARCH:?set by the Makefile}"
: "${GPUSOLVER2_MPI_BIN:?set by the Makefile}"
: "${GPUSOLVER2_MPI_LIBDIR:?set by the Makefile}"
: "${GPUSOLVER2_ROCM:?set by the Makefile}"
: "${GPUSOLVER2_AOCL_LIB:?set by the Makefile}"
: "${GPUSOLVER2_AOCL_INC:?set by the Makefile}"
: "${GPUSOLVER2_AOCC_LIB:?set by the Makefile}"
: "${GPUSOLVER2_SCALAPACK_SO:?set by the Makefile}"
: "${GPUSOLVER2_ELPA_VER:=2026.02.002}"
: "${GPUSOLVER2_CMAKE:=}"
# Match the CUDA builder: a private Autoconf installation may be required
# on compute systems whose /usr/bin version predates the pinned ELPA.
: "${GPUSOLVER2_TOOL_PATH:=}"
: "${GPUSOLVER2_JOBS:=$(nproc 2>/dev/null || echo 8)}"

# The CMake configuration carries resolved MPI wrappers and LP64 providers.
# Native make without this file continues to use its AOCL/OpenMPI defaults.
if [ -n "${GPUSOLVER2_HOST_CONFIG:-}" ]; then
  . "$GPUSOLVER2_HOST_CONFIG"
fi
: "${GPUSOLVER2_HOST_FC:=$GPUSOLVER2_ROCM/llvm/bin/flang}"
: "${GPUSOLVER2_MPICC:=$GPUSOLVER2_MPI_BIN/mpicc}"
: "${GPUSOLVER2_MPICXX:=$GPUSOLVER2_MPI_BIN/mpicxx}"
: "${GPUSOLVER2_MPIFC:=$GPUSOLVER2_MPI_BIN/mpif90}"
: "${GPUSOLVER2_HOST_MATH:=-L$GPUSOLVER2_AOCL_LIB -lscalapack -lflame -lblis -laoclutils}"
: "${GPUSOLVER2_HOST_BLAS:=$GPUSOLVER2_AOCL_LIB/libblis.so}"
: "${GPUSOLVER2_HOST_SCALAPACK:=$GPUSOLVER2_SCALAPACK_SO}"
: "${GPUSOLVER2_HOST_LDFLAGS:=-L$GPUSOLVER2_MPI_LIBDIR -L$GPUSOLVER2_AOCL_LIB -L$GPUSOLVER2_AOCC_LIB -L$GPUSOLVER2_ROCM/lib}"
: "${GPUSOLVER2_RUNTIME_PATH:=$GPUSOLVER2_ROCM/llvm/lib:$GPUSOLVER2_MPI_LIBDIR:$GPUSOLVER2_ROCM/lib:$GPUSOLVER2_AOCL_LIB:$GPUSOLVER2_AOCC_LIB}"
: "${GPUSOLVER2_ELPA_MPI_OPTION:=}"

DIST=$GPUSOLVER2_DIST
SRC=$GPUSOLVER2_BUILD/src
BLD=$GPUSOLVER2_BUILD/build
STAMP=$GPUSOLVER2_BUILD/stamp
LOGS=$GPUSOLVER2_BUILD/logs
P=$GPUSOLVER2_PREFIX
J=$GPUSOLVER2_JOBS
mkdir -p "$SRC" "$BLD" "$STAMP" "$LOGS" "$P"

# Compiler, MPI, GPU target and host-math paths are embedded in the generated
# configure/CMake caches and objects. Resume only when those inputs match.
# Keep extracted sources, the installed prefix and bootstrapped cmake: the
# rebuilt ELPA/COSMA install replaces their own artifacts after a successful
# build, without deleting unrelated files in a user-supplied prefix.
printf '%s\n' "$GPUSOLVER2_HIPCC" "$GPUSOLVER2_GPU_ARCH" \
  "$GPUSOLVER2_MPI_BIN" "$GPUSOLVER2_MPI_LIBDIR" "$GPUSOLVER2_ROCM" \
  "$GPUSOLVER2_AOCL_LIB" "$GPUSOLVER2_AOCL_INC" "$GPUSOLVER2_AOCC_LIB" \
  "$GPUSOLVER2_SCALAPACK_SO" "$GPUSOLVER2_ELPA_VER" "$GPUSOLVER2_HOST_CC" \
  "$GPUSOLVER2_HOST_CXX" "$GPUSOLVER2_PREFIX" "$GPUSOLVER2_DIST" \
  "$GPUSOLVER2_CMAKE" "$GPUSOLVER2_TOOL_PATH" > "$STAMP/toolchain.tmp"
# A build-script fix must also invalidate successful steps from older rules.
cksum < "$0" >> "$STAMP/toolchain.tmp"
if [ -n "${GPUSOLVER2_HOST_CONFIG:-}" ]; then
  cksum < "$GPUSOLVER2_HOST_CONFIG" >> "$STAMP/toolchain.tmp"
fi
if ! cmp -s "$STAMP/toolchain.tmp" "$STAMP/toolchain.conf"; then
  rm -f "$STAMP/01_elpa.done" "$STAMP/02_cosma.done"
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
export PATH="${GPUSOLVER2_TOOL_PATH:+$GPUSOLVER2_TOOL_PATH:}$GPUSOLVER2_MPI_BIN:$GPUSOLVER2_ROCM/bin:$GPUSOLVER2_ROCM/llvm/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$GPUSOLVER2_RUNTIME_PATH"

fail() { echo "gpusolver2 stack: FAILED at step $1 (see $LOGS/$1.log)"; exit 1; }

run_step() {
  step=$1; shift
  [ -f "$STAMP/$step.done" ] && return 0
  echo "gpusolver2 stack: $step"
  # set -e inside the subshell: without it a step "succeeds" when merely its
  # LAST command does (e.g. a trailing rm -f), stamping a half-built step.
  # The status must be tested on a separate line -- POSIX ignores set -e
  # everywhere inside the left side of an || list.
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
  # An existing configure script can belong to a different ELPA release.
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
    ( cd "$SRC" && unzip -q "$DIST/elpa-$ELPA_TAG.zip" && mv elpa-"$ELPA_TAG"-* elpa )
    ( cd "$SRC/elpa" && sh autogen.sh )
    printf '%s\n' "$GPUSOLVER2_ELPA_VER" > "$SRC/elpa/.openmx-source-version"
  fi
  # ROCm Flang passes a type(c_ptr) output argument to these duplicate
  # bind(C) interfaces as its value instead of its address. An initially
  # null C pointer then reaches hipMalloc/hipHostMalloc as a null output
  # slot. Use the working intptr interface and transfer the returned bits.
  # Apply on every ELPA build, including sources retained after a stamp
  # invalidation; reject unexpected upstream changes instead of mispatching.
  python3 - "$SRC/elpa/src/GPU/ROCm/hip_template.F90" <<'OPENMX_ELPA_CPTR_PATCH'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
patched = source
for host in ("", "host_"):
    original = f"""    function hip_malloc_{host}cptr(a, width_height) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                               :: a
      integer(kind=c_intptr_t), intent(in)      :: width_height
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_malloc_{host}cptr_c(a, width_height) /= 0
#else
      success = .true.
#endif
    end function"""
    replacement = original.replace(
        f"      success = hip_malloc_{host}cptr_c(a, width_height) /= 0",
        "      integer(kind=c_intptr_t) :: address\n"
        "      ! Keep pointer output storage intact across the Flang C binding.\n"
        "      address = 0_c_intptr_t\n"
        f"      success = hip_malloc_{host}intptr_c(address, width_height) /= 0\n"
        "      if (success) a = transfer(address, a)",
    )
    old_count = patched.count(original)
    new_count = patched.count(replacement)
    if old_count == 1 and new_count == 0:
        patched = patched.replace(original, replacement, 1)
    elif old_count != 0 or new_count != 1:
        raise SystemExit(
            f"ERROR: unsupported ELPA hip_malloc_{host}cptr source in {path} "
            f"(original={old_count}, patched={new_count})"
        )
if patched != source:
    path.write_text(patched)
OPENMX_ELPA_CPTR_PATCH
  mkdir -p "$BLD/elpa" && cd "$BLD/elpa"
  # -I.../include/rocsolver: ROCm >= 6 nests rocsolver.h in its own
  # subdirectory but ELPA's ROCm sources include plain <rocsolver.h>.
  # The SSE/AVX/AVX2 kernels stay enabled (their default) and the whole
  # library is built with the x86-64-v3 baseline so ELPA's intrinsics
  # probes pass; openmx_elpa_cpu.c picks the fastest compiled kernel the
  # CPU supports at run time (ELPA itself checks the CPU only with
  # heterogeneous-cluster support), which is what the CPU-kernel fallback
  # of the memory guard runs on.  AVX-512 is
  # disabled: it would raise the baseline beyond Zen 2 build/dev hosts, and
  # on the Zen 4 target its double-pumped AVX-512 gains little over AVX2.
  # (The NVIDIA-side script disables all SIMD kernels for its NVHPC
  # toolchain.)
  "$SRC/elpa/configure" --prefix="$P" \
    FC="$GPUSOLVER2_MPIFC" CC="$GPUSOLVER2_MPICC" CXX="$GPUSOLVER2_MPICXX" \
    HIPCC="$GPUSOLVER2_HIPCC" \
    FCFLAGS="-O2 -march=x86-64-v3" CFLAGS="-O2 -march=x86-64-v3" CXXFLAGS="-O2 -march=x86-64-v3" \
    HIPCCFLAGS="-O2 --offload-arch=$GPUSOLVER2_GPU_ARCH -I$GPUSOLVER2_ROCM/include/rocsolver" \
    CPPFLAGS="-I$GPUSOLVER2_ROCM/include -I$GPUSOLVER2_ROCM/include/rocsolver" \
    LDFLAGS="$GPUSOLVER2_HOST_LDFLAGS" \
    LIBS="$GPUSOLVER2_HOST_MATH -lrocblas -lamdhip64 -lstdc++" \
    --enable-amd-gpu-kernels --disable-shared --enable-static $GPUSOLVER2_ELPA_MPI_OPTION \
    --enable-generic-kernels --disable-avx512-kernels \
    --disable-c-tests --disable-cpp-tests --disable-fortran-tests --disable-Fortran-tests
  # amdflang's -O2 middle end blows up (tens of GiB of RSS) on the
  # generated elpa2_compute.F90 once the SIMD kernel templates are enabled;
  # build that one driver TU at -O1 up front.  The hot kernels live in
  # their own TUs and keep the full optimization level.
  make FCFLAGS="-O1 -march=x86-64-v3" src/elpa2/libelpa_private_la-elpa2_compute.lo
  # bin_PROGRAMS= noinst_PROGRAMS=: skip the elpa2_print_kernels diagnostic
  # binary and the validate_* self-test binaries ("make all" builds both
  # even with the test suites disabled).  Their libtool links expand
  # OpenMPI's .la dependency_libs (-levent_core, -lhwloc) whose dev
  # symlinks plain distro hosts do not ship; nothing in the gpusolver2
  # stack runs them.  The noinst override also drops libelpatest (its
  # Fortran-module build order is racy and only the skipped validate
  # binaries link it) while keeping the two convenience libraries that
  # make up libelpa itself.
  make -j "$J" bin_PROGRAMS= noinst_PROGRAMS= \
    noinst_LTLIBRARIES="libelpa_public.la libelpa_private.la"
  make install bin_PROGRAMS= noinst_PROGRAMS= \
    noinst_LTLIBRARIES="libelpa_public.la libelpa_private.la"
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

# The device code of COSMA's Tiled-MM backend is ordinary C++ over the HIP
# API, so the whole C++ side is compiled with hipcc (clang); the C side stays
# uses the selected host compiler.
COMMON="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P \
 -DCMAKE_INSTALL_LIBDIR=lib \
 -DCMAKE_PREFIX_PATH=$P;$GPUSOLVER2_ROCM \
 -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
 -DCMAKE_C_COMPILER=$GPUSOLVER2_HOST_CC -DCMAKE_CXX_COMPILER=$GPUSOLVER2_HIPCC \
 -DCMAKE_Fortran_COMPILER=$GPUSOLVER2_HOST_FC \
 -DCMAKE_CXX_FLAGS=--offload-arch=$GPUSOLVER2_GPU_ARCH \
 -DCMAKE_HIP_ARCHITECTURES=$GPUSOLVER2_GPU_ARCH \
 -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
MPIARGS="-DMPI_C_COMPILER=$GPUSOLVER2_MPICC -DMPI_CXX_COMPILER=$GPUSOLVER2_MPICXX \
 -DMPI_Fortran_COMPILER=$GPUSOLVER2_MPIFC -DMPI_CXX_SKIP_MPICXX=ON"

# Keep MPI discovery on the selected host compiler and HIP C++ toolchain.
export OMPI_CC=$GPUSOLVER2_HOST_CC
export OMPI_CXX=$GPUSOLVER2_HIPCC

# ---------- COSMA (communication-optimal pdgemm/pzgemm, ROCm backend;
#            COSTA and Tiled-MM are provided offline at the pinned commits)
step_cosma() {
  [ -d "$SRC/cosma" ] || { cd "$SRC" && unzip -q "$DIST/cosma-v$V_COSMA.zip" && mv "COSMA-$V_COSMA" cosma; }
  [ -d "$SRC/COSTA" ] || { cd "$SRC" && unzip -q "$DIST/COSTA-$SHA_COSTA.zip" && mv "COSTA-$SHA_COSTA" COSTA; }
  [ -d "$SRC/Tiled-MM" ] || { cd "$SRC" && unzip -q "$DIST/Tiled-MM-$SHA_TILEDMM.zip" && mv "Tiled-MM-$SHA_TILEDMM" Tiled-MM; }
  # COSMA_SCALAPACK=CUSTOM forces automatic host-BLAS detection. Skip its
  # independent MKL/MPI probe and supply the already resolved host BLAS via
  # its OPENBLAS target (also for AOCL and MKL); no second vendor is selected.
  $CMAKE -S "$SRC/cosma" -B "$BLD/cosma" $COMMON $MPIARGS \
    -DCOSMA_BLAS=ROCM -DCOSMA_SCALAPACK=CUSTOM -DCMAKE_DISABLE_FIND_PACKAGE_MKL=ON \
    -DCOSMA_SCALAPACK_LINK_LIBRARIES="$GPUSOLVER2_HOST_SCALAPACK" \
    -DCOSMA_OPENBLAS_LINK_LIBRARIES="$GPUSOLVER2_HOST_BLAS" \
    -DCOSMA_OPENBLAS_INCLUDE_DIRS="" \
    -DCOSMA_WITH_TESTS=OFF -DCOSMA_WITH_APPS=OFF -DCOSMA_WITH_BENCHMARKS=OFF \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_COSTA="$SRC/COSTA" \
    "-DFETCHCONTENT_SOURCE_DIR_TILED-MM=$SRC/Tiled-MM"
  $CMAKE --build "$BLD/cosma" -j "$J"
  $CMAKE --install "$BLD/cosma"
}
run_step 02_cosma step_cosma

echo "gpusolver2 stack: all components installed into $P"
