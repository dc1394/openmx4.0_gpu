# Shared build settings for standalone HIP fixtures. Source from a runner.
# GPU_ARCH should match the allocation (gfx942 for MI300A, gfx1200 for RX 9060 XT).
rocm=${ROCM:-${HIP_PATH:-/opt/rocm-7.2.0}}
hipcc=${HIPCC:-$rocm/bin/hipcc}
cc=${CC:-$rocm/llvm/bin/clang}
gpu_arch=${GPU_ARCH:-gfx942}
aocl=${AOCL_HOME:-/system/apps/rhel/aocl/5.1.0/aocc}
aocc_lib=${AOCC_LIB:-/system/apps/rhel/aocc/5.0.0/lib}
export LD_LIBRARY_PATH="$rocm/lib:$rocm/llvm/lib:$aocl/lib:$aocc_lib:${LD_LIBRARY_PATH:-}"
