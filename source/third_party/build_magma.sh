#!/bin/sh
# Optional, pinned MAGMA dependency. No tracked archive or generated file is
# needed: downloads/builds stay in source/build/magma, install is local.
# Required: MAGMA_CUDA_HOME, MAGMA_GPU_ARCH and LP64 MAGMA_LAPACK_LIBRARIES
# (a semicolon-separated CMake list). MAGMA_MATH_DIR supports split NVHPC SDKs.
# Override MAGMA_CMAKE, MAGMA_CC, MAGMA_CXX, MAGMA_NVCC, MAGMA_HOST_LIBDIR,
# MAGMA_RUNTIME_DIRS (semicolon list), MAGMA_JOBS, MAGMA_BUILD/MAGMA_PREFIX.
# MAGMA_ARCHIVE may provide an offline copy; its hash is always verified.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
version=2.10.0
digest=ea0c57fcb64ac2fd7ffe8f02d8fe18f07055c5b7fba0164f565d1e3a85148fb5
build_root=${MAGMA_BUILD:-$source_dir/build/magma}
prefix=${MAGMA_PREFIX:-$script_dir/magma-install}
cmake=${MAGMA_CMAKE:-cmake}
cc=${MAGMA_CC:-gcc}
cxx=${MAGMA_CXX:-g++}
: "${MAGMA_CUDA_HOME:?set MAGMA_CUDA_HOME to the selected CUDA toolkit}"
: "${MAGMA_GPU_ARCH:?set MAGMA_GPU_ARCH (for example 80)}"
: "${MAGMA_LAPACK_LIBRARIES:?set MAGMA_LAPACK_LIBRARIES to LP64 LAPACK and BLAS libraries}"
math_dir=${MAGMA_MATH_DIR:-$MAGMA_CUDA_HOME}
math_libdir=
for candidate in "$math_dir/lib64" "$math_dir/lib"; do
    if [ -f "$candidate/libcublas.so" ] && [ -f "$candidate/libcusparse.so" ]; then
        math_libdir=$candidate
        break
    fi
done
if [ -z "$math_libdir" ]; then
    echo "Cannot find libcublas.so and libcusparse.so under $math_dir/lib64 or $math_dir/lib" >&2
    exit 2
fi
nvcc=${MAGMA_NVCC:-$MAGMA_CUDA_HOME/bin/nvcc}
jobs=${MAGMA_JOBS:-3}
case $jobs in ''|*[!0-9]*|0) echo 'MAGMA_JOBS must be positive' >&2; exit 2;; esac
mkdir -p "$build_root" "$prefix"
build_root=$(CDPATH= cd -- "$build_root" && pwd)
prefix=$(CDPATH= cd -- "$prefix" && pwd)
archive=${MAGMA_ARCHIVE:-$build_root/magma-$version.tar.gz}
if [ ! -f "$archive" ]; then
    if [ -n "${MAGMA_ARCHIVE:-}" ]; then echo "Missing MAGMA_ARCHIVE: $archive" >&2; exit 2; fi
    curl -fL --retry 2 "https://icl.utk.edu/projectsfiles/magma/downloads/magma-$version.tar.gz" -o "$archive.tmp"
    mv "$archive.tmp" "$archive"
fi
printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
if [ ! -d "$build_root/magma-$version" ]; then
    tar -xzf "$archive" -C "$build_root"
fi
# The one-stage drivers use logical IDs 0..ngpu-1. This exact private patch
# maps those IDs to the selected Band group without changing CUDA visibility.
# Forward/reverse dry-runs reject an unexpected source tree and make repeat
# builds idempotent. Only the CUDA interface translation unit is changed.
map_patch="$script_dir/magma-device-map.patch"
map_header="$script_dir/magma_device_map.h"
map_interface="$build_root/magma-$version/interface_cuda/interface.cpp"
map_original=4c52ae562ea1feafe9cbf1e93670d336215f74871369fb05072f457a6c1e4f09
map_patched=b5d14eaeaef28e4d1ffcfb530e7b9dab50fb5df3f018f765eb3191e9ff3d3a80
if ! printf '%s  %s\n' "$map_original" "$map_interface" | sha256sum -c - >/dev/null 2>&1 &&
   ! printf '%s  %s\n' "$map_patched" "$map_interface" | sha256sum -c - >/dev/null 2>&1; then
    echo 'Unexpected MAGMA CUDA interface hash; refusing to patch' >&2
    exit 2
fi
if patch --batch --fuzz=0 --forward --dry-run -d "$build_root/magma-$version" -p1 < "$map_patch" >/dev/null 2>&1; then
    patch --batch --fuzz=0 --forward -d "$build_root/magma-$version" -p1 < "$map_patch"
elif ! patch --batch --fuzz=0 --reverse --dry-run -d "$build_root/magma-$version" -p1 < "$map_patch" >/dev/null 2>&1; then
    echo 'MAGMA source does not match the pinned device-map patch' >&2
    exit 2
fi
printf '%s  %s\n' "$map_patched" "$map_interface" | sha256sum -c -
if ! cmp -s "$map_header" "$build_root/magma-$version/interface_cuda/magma_device_map.h"; then
    cp "$map_header" "$build_root/magma-$version/interface_cuda/magma_device_map.h"
fi
runtime_dirs="$math_dir/lib64;$math_dir/lib;$MAGMA_CUDA_HOME/lib64;${MAGMA_RUNTIME_DIRS:-}"
if [ -n "${MAGMA_HOST_LIBDIR:-}" ]; then runtime_dirs="$runtime_dirs;$MAGMA_HOST_LIBDIR"; fi
# CMake's CUDA discovery needs explicit math-library paths for split NVHPC.
# Build only libmagma, not the thousands of optional testers/sparse targets.
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
"$cmake" -S "$build_root/magma-$version" -B "$build_root/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_HOST_COMPILER="$cxx" \
    -DCUDAToolkit_ROOT="$MAGMA_CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES="$MAGMA_GPU_ARCH" \
    -DMAGMA_ENABLE_CUDA=ON -DUSE_FORTRAN=OFF -DBUILD_SHARED_LIBS=ON \
    -DLAPACK_LIBRARIES="$MAGMA_LAPACK_LIBRARIES" \
    -DCUDA_cublas_LIBRARY="$math_libdir/libcublas.so" \
    -DCUDA_cusparse_LIBRARY="$math_libdir/libcusparse.so" \
    -DCMAKE_CXX_FLAGS="-I$math_dir/include" -DCMAKE_CUDA_FLAGS="-I$math_dir/include" \
    -DCMAKE_BUILD_RPATH="$runtime_dirs" -DCMAKE_INSTALL_RPATH="$runtime_dirs" \
    -DCMAKE_INSTALL_PREFIX="$prefix"
"$cmake" --build "$build_root/build" --target magma --parallel "$jobs"
mkdir -p "$prefix/lib" "$prefix/include" "$prefix/share/magma"
# Replace each library atomically: an already running OpenMX process may
# still map the previous inode while this optional dependency is rebuilt.
for artifact in "$build_root/build/lib/libmagma.so"*; do
    artifact_name=${artifact##*/}
    temporary="$prefix/lib/.$artifact_name.tmp.$$"
    cp -P "$artifact" "$temporary"
    mv -f "$temporary" "$prefix/lib/$artifact_name"
done
cp "$build_root/magma-$version/include/"*.h "$prefix/include/"
if [ -d "$build_root/build/include" ]; then cp "$build_root/build/include/"*.h "$prefix/include/"; fi
cp "$build_root/magma-$version/COPYRIGHT" "$prefix/share/magma/COPYRIGHT"
printf 'MAGMA %s\narchive_sha256=%s\nCUDA=%s\nGPU_ARCH=%s\n' \
    "$version" "$digest" "$MAGMA_CUDA_HOME" "$MAGMA_GPU_ARCH" > "$prefix/share/magma/build.txt"
sha256sum "$map_patch" "$map_header" >> "$prefix/share/magma/build.txt"
printf 'MAGMA installed in %s\n' "$prefix"
