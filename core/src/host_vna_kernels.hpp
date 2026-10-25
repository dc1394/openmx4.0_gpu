#ifndef OPENMX_HOST_VNA_KERNELS_HPP
#define OPENMX_HOST_VNA_KERNELS_HPP
#include "openmx_core_vna_simd.h"
#include "host_simd_dispatch.hpp"

// Private, ordinary C++ boundary of the Set_ProExpn_VNA family. No Kokkos
// vector type or ISA flag crosses into the C ABI or device translation units.
namespace openmx::core::host {
struct VnaKernels {
  int width;
  void (*contract)(const OpenMXVnaContract&);
  void (*bessel)(const OpenMXVnaBessel&);
  void (*quadrature)(const OpenMXVnaQuadrature&);
  void (*hvna_block)(const OpenMXVnaHvnaBlock&);
  void (*angular)(const OpenMXVnaAngular&);
  void (*convert)(const OpenMXVnaConvert&);
};
extern "C" OPENMX_HOST_SIMD_EXPORT const VnaKernels* openmx_host_vna_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(VnaKernels,vna)
}
#endif
