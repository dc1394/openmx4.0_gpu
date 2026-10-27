#!/bin/bash
# Standalone accuracy check of HIP CSR and on-the-fly density kernels.
# No OpenMX build is needed. Run alone on the GPU for useful timings.
# ROCM/ARCH select the SDK/device; OUT retains the executable and build log.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
if [ -f "$root/source/openmx_env.c" ]; then
    source_dir="$root/source"
    kernel_dir="$source_dir"
else
    source_dir=$(cd "$root/../.." && pwd)
    kernel_dir="$root/src"
fi
ROCM=${ROCM:-/opt/rocm-7.2.0}
ARCH=${ARCH:-gfx1200}
OUT=${OUT:-$root/work/check_density_df_hip}
mkdir -p "$OUT"
for name in openmx_env openmx_precision; do
    "${CC:-$ROCM/llvm/bin/clang}" -std=c11 -O2 -I"$source_dir" \
        -c "$source_dir/$name.c" -o "$OUT/$name.o"
done
# Compile and link separately: hipcc can retain '-x hip' after a .cu input.
"$ROCM/bin/hipcc" -std=c++20 -O3 --offload-arch="$ARCH" -ffp-contract=off \
    -DOCML_BASIC_ROUNDED_OPERATIONS -Wno-unused-result -Wno-unused-value \
    -I"$kernel_dir" -I"$source_dir" -c "$root/tools/check_density_df_hip.cu" \
    -o "$OUT/check_density_df_hip.o" > "$OUT/build.log" 2>&1 ||
    { cat "$OUT/build.log"; exit 1; }
"$ROCM/bin/hipcc" --offload-arch="$ARCH" "$OUT/check_density_df_hip.o" \
    "$OUT/openmx_env.o" "$OUT/openmx_precision.o" -o "$OUT/check_density_df_hip" \
    >> "$OUT/build.log" 2>&1 || { cat "$OUT/build.log"; exit 1; }
"$OUT/check_density_df_hip"
