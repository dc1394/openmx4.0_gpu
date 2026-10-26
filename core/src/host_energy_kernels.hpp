#ifndef OPENMX_HOST_ENERGY_KERNELS_HPP
#define OPENMX_HOST_ENERGY_KERNELS_HPP
#include "openmx_core_energy_simd.h"
#include "host_simd_dispatch.hpp"

// Private, ordinary C++ boundary of the energy family (neutral-atom
// two-centre integrand, band density-matrix blocks). No Kokkos vector type or
// ISA flag crosses into the C ABI or device translation units.
namespace openmx::core::host {
struct EnergyKernels {
  int width;
  void (*neutral_points)(const OpenMXEnergyNeutralPoints&);
  void (*neutral_sum)(const OpenMXEnergyNeutralPoints&,double*,double*);
  void (*xc_shell)(const OpenMXEnergyXcShell&);
  void (*band_block)(const OpenMXEnergyBandBlock&);
};
extern "C" OPENMX_HOST_SIMD_EXPORT const EnergyKernels* openmx_host_energy_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(EnergyKernels,energy)
}
#endif
