#include "host_grid_kernels.hpp"
#include <cstring>

namespace openmx::core::host {
const GridKernels* openmx_host_grid_kernels(int mode) noexcept {
  if (mode == OPENMX_GRID_SCALAR) return &grid_scalar();
#if defined(OPENMX_HOST_SIMD_AVX2) || defined(OPENMX_HOST_SIMD_AVX512)
  __builtin_cpu_init();
#ifdef OPENMX_HOST_SIMD_AVX2
  if (mode == OPENMX_GRID_AVX2 && __builtin_cpu_supports("avx2"))
    return &grid_avx2();
#endif
#ifdef OPENMX_HOST_SIMD_AVX512
  if (mode == OPENMX_GRID_AVX512 && __builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl"))
    return &grid_avx512();
#endif
#endif
#ifdef OPENMX_HOST_SIMD_NEON
  if (mode == OPENMX_GRID_NEON) return &grid_neon();
#endif
  return nullptr;
}
int openmx_host_grid_mode(const char* value) noexcept {
  // Retain the legacy environment parser's false/no/0 spellings.
  if (value && (value[0]=='0' || value[0]=='n' || value[0]=='N' ||
                value[0]=='f' || value[0]=='F' || !std::strcmp(value,"scalar") ||
                !std::strcmp(value,"SCALAR"))) return OPENMX_GRID_SCALAR;
  const bool cap_avx2=value && (!std::strcmp(value,"avx2") || !std::strcmp(value,"AVX2"));
  if (!cap_avx2 && openmx_host_grid_kernels(OPENMX_GRID_AVX512)) return OPENMX_GRID_AVX512;
  if (openmx_host_grid_kernels(OPENMX_GRID_AVX2)) return OPENMX_GRID_AVX2;
  if (openmx_host_grid_kernels(OPENMX_GRID_NEON)) return OPENMX_GRID_NEON;
  return OPENMX_GRID_SCALAR;
}
}
