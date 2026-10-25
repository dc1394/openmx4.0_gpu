#ifndef OPENMX_HOST_ORBITALS_KERNELS_HPP
#define OPENMX_HOST_ORBITALS_KERNELS_HPP
#include "openmx_core_orbitals_simd.h"
#include "host_simd_dispatch.hpp"
#include <cstddef>

// Private, ordinary C++ boundary of the orbital-grid family. No Kokkos
// vector type or ISA flag crosses into the C ABI or device translation units.
// The kernel assumes the shapes validated by openmx_core_orbitals_points.
namespace openmx::core::host {
struct OrbitalsKernels {
  int width;
  void (*points)(const OpenMXOrbitalsTables&,const OpenMXOrbitalsFrame&,const OpenMXOrbitalsPoint*,size_t,float*);
};
extern "C" OPENMX_HOST_SIMD_EXPORT const OrbitalsKernels* openmx_host_orbitals_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OrbitalsKernels,orbitals)
}
#endif
