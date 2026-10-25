#include "host_simd_dispatch.hpp"
#include "host_olg_kernels.hpp"
#include "host_vna_kernels.hpp"
#include "host_force_kernels.hpp"
#include "host_orbitals_kernels.hpp"
#include "host_energy_kernels.hpp"

namespace openmx::core::host {
extern "C" int openmx_host_simd_usable(int mode) noexcept {
  if(mode==OPENMX_GRID_SCALAR)return 1;
#if defined(OPENMX_HOST_SIMD_AVX2) || defined(OPENMX_HOST_SIMD_AVX512)
  __builtin_cpu_init();
#ifdef OPENMX_HOST_SIMD_AVX2
  if(mode==OPENMX_GRID_AVX2)return __builtin_cpu_supports("avx2")?1:0;
#endif
#ifdef OPENMX_HOST_SIMD_AVX512
  if(mode==OPENMX_GRID_AVX512)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl")?1:0;
#endif
#endif
#ifdef OPENMX_HOST_SIMD_NEON
  if(mode==OPENMX_GRID_NEON)return 1;
#endif
  return 0;
}
OPENMX_HOST_SIMD_DISPATCHER(OlgKernels,olg)
OPENMX_HOST_SIMD_DISPATCHER(VnaKernels,vna)
OPENMX_HOST_SIMD_DISPATCHER(OrbitalsKernels,orbitals)
OPENMX_HOST_SIMD_DISPATCHER(EnergyKernels,energy)
OPENMX_HOST_SIMD_DISPATCHER(ForceKernels,force)
}
