#!/bin/sh
set -eu

# Build tests/gemmul8_reuse_probe.cu into <bindir> and, when probe arguments
# follow, run it.  Two binaries are kept: gemmul8_reuse_probe for wall times
# and gemmul8_reuse_probe_prof, linked against a GEMMul8_PROFILE=1 object,
# for GEMMul8's internal phase times (select it with PROBE=prof).  The
# GEMMul8 translation unit takes several minutes to compile; existing
# binaries are reused, so a build on a login node serves later GPU jobs.
#   tests/run_gemmul8_reuse_probe.sh <bindir>                  (build only)
#   tests/run_gemmul8_reuse_probe.sh <bindir> capacity 5000 13000
#   tests/run_gemmul8_reuse_probe.sh <bindir> run --x X_g1.bin --h H_g1_scf001_s0.bin
# Toolkit selection as in tests/run_gemmul8_smoke.sh; GPU_ARCH must be set
# where nvidia-smi is missing.  GEMMUL8_TEST_OBJECT names an existing object
# of source/gemmul8_openmx.cu built without profiling, HOST_LIBDIR the
# run-time library directory of a host compiler that is not the system one.
# The double-double reference of the probe needs a host compiler that does
# not contract a*b+c on its own: GCC gets -ffp-contract=off, which nvc++
# does not know (set NVCC_HOST=g++ where possible).  "selftest" tells whether
# the reference arithmetic survived the compiler; PROBE_HOST_FLAGS replaces
# the host flags.
[ $# -ge 1 ] || { echo "usage: $0 <bindir> [probe arguments]" >&2; exit 2; }
bin_dir=$1
shift
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda-13.4}
math_root=${CUDA_MATH_ROOT:-$cuda_root}
sdk_root=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.9}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
host_cxx=${NVCC_HOST:-$sdk_root/compilers/bin/nvc++}
gemmul8_root=${GEMMUL8_DIR:-$test_root/source/third_party/GEMMul8}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS

mkdir -p "$bin_dir"
bin_dir=$(CDPATH= cd -- "$bin_dir" && pwd)

if [ ! -x "$bin_dir/gemmul8_reuse_probe" ] || [ ! -x "$bin_dir/gemmul8_reuse_probe_prof" ] ||
   [ "$test_root/tests/gemmul8_reuse_probe.cu" -nt "$bin_dir/gemmul8_reuse_probe" ]; then
    gpu_arch=${GPU_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '/^[0-9]+[.][0-9]+$/ {gsub(/[.]/, ""); print; exit}')}
    rpath="-Xlinker -rpath -Xlinker $math_root/lib64 -Xlinker -rpath -Xlinker $cuda_root/lib64"
    [ -z "${HOST_LIBDIR:-}" ] || rpath="$rpath -Xlinker -rpath -Xlinker $HOST_LIBDIR"

    object() {  # object <output> [extra nvcc options]
        output=$1
        shift
        [ -f "$output" ] && [ "$output" -nt "$test_root/source/gemmul8_openmx.cu" ] && return 0
        "$nvcc" -ccbin "$host_cxx" -std=c++20 -O3 -diag-suppress 177 "$@" \
            -DGPU_ARCH="$gpu_arch" -arch="sm_$gpu_arch" \
            -I"$gemmul8_root/include" -I"$gemmul8_root/src" -I"$math_root/include" \
            -c "$test_root/source/gemmul8_openmx.cu" -o "$output"
    }
    if [ -n "${PROBE_HOST_FLAGS:-}" ]; then
        host_flags=$PROBE_HOST_FLAGS
    elif "$host_cxx" --version 2>/dev/null | grep -q -i -E 'nvc\+\+|nvidia|pgi'; then
        host_flags="-mp"
    else
        host_flags="-fopenmp -mfma -ffp-contract=off"
    fi
    xcompiler=
    for flag in $host_flags; do xcompiler="$xcompiler -Xcompiler $flag"; done
    probe() {  # probe <output> <GEMMul8 object>
        "$nvcc" -ccbin "$host_cxx" -std=c++20 -O2 -arch="sm_$gpu_arch" $xcompiler \
            -I"$gemmul8_root/include" -I"$math_root/include" \
            "$test_root/tests/gemmul8_reuse_probe.cu" "$2" \
            -L"$math_root/lib64" -L"$cuda_root/lib64" -lcublas -lcublasLt $rpath -o "$1"
    }

    plain_object=${GEMMUL8_TEST_OBJECT:-$bin_dir/gemmul8.o}
    [ -n "${GEMMUL8_TEST_OBJECT:-}" ] || object "$plain_object"
    object "$bin_dir/gemmul8_profile.o" -DGEMMul8_PROFILE=1
    probe "$bin_dir/gemmul8_reuse_probe" "$plain_object"
    probe "$bin_dir/gemmul8_reuse_probe_prof" "$bin_dir/gemmul8_profile.o"
fi

[ $# -ge 1 ] || exit 0
if [ "${PROBE:-}" = prof ]; then
    exec "$bin_dir/gemmul8_reuse_probe_prof" "$@"
fi
exec "$bin_dir/gemmul8_reuse_probe" "$@"
