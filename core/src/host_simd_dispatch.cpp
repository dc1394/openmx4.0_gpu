#include "host_simd_dispatch.hpp"
#include "host_olg_kernels.hpp"
#include "host_vna_kernels.hpp"
#include "host_force_kernels.hpp"
#include "host_orbitals_kernels.hpp"
#include "host_energy_kernels.hpp"
#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#endif

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
#if defined(__linux__) && defined(__aarch64__)
  int required=0;
#ifdef OPENMX_HOST_SIMD_SVE128
  if(mode==OPENMX_GRID_SVE128)required=16;
#endif
#ifdef OPENMX_HOST_SIMD_SVE256
  if(mode==OPENMX_GRID_SVE256)required=32;
#endif
#ifdef OPENMX_HOST_SIMD_SVE512
  if(mode==OPENMX_GRID_SVE512)required=64;
#endif
  // Linux tracks VL per thread. Never cache it or change the caller's VL.
  // A wider current VL is unsafe too: these objects use fixed-size vectors.
  if(required && (getauxval(AT_HWCAP)&HWCAP_SVE)) {
    const int vl=prctl(PR_SVE_GET_VL,0,0,0,0);
    return vl>=0 && (vl&PR_SVE_VL_LEN_MASK)==required;
  }
#endif
  return 0;
}
OPENMX_HOST_SIMD_DISPATCHER(OlgKernels,olg)
OPENMX_HOST_SIMD_DISPATCHER(VnaKernels,vna)
OPENMX_HOST_SIMD_DISPATCHER(OrbitalsKernels,orbitals)
OPENMX_HOST_SIMD_DISPATCHER(EnergyKernels,energy)
OPENMX_HOST_SIMD_DISPATCHER(ForceKernels,force)
}
