#ifndef OPENMX_HOST_FORCE_KERNELS_HPP
#define OPENMX_HOST_FORCE_KERNELS_HPP
#include "openmx_core_force_simd.h"
#include "host_simd_dispatch.hpp"
#include <cstddef>

// Private, ordinary C++ boundary of the force family. The C ABI layer
// validates every request; the kernels trust their arguments. Scratch sizes
// are fixed by the API layer (pad = OPENMX_FORCE_SIMD_PAD elements) so that a
// request validated once serves every ISA.
namespace openmx::core::host {
constexpr int force_simd_pad=OPENMX_FORCE_SIMD_PAD;
inline size_t force_padded(int n) { return (size_t(n)+force_simd_pad-1)/force_simd_pad*force_simd_pad; }
// Planes of the lane operand of a VNA block: value(+column derivatives) when
// the lanes span the columns, derivatives(+row value) when they span the rows.
inline int force_vna_planes(int terms,int lanes_rows) { return terms==2?4:(lanes_rows?3:1); }
struct ForceKernels {
  int width;
  void (*vna_lanes)(const float *const *const *rows,int components,int count,int projectors,float *planes,size_t pad);
  void (*vna_block)(const OpenMXForceVnaBlock&,void *scratch);
  void (*grid_pair)(const OpenMXForceGridPair&,double *sums,void *scratch);
  void (*grid_density)(const double *const *density[4],int rows,int columns,int columns_max,int spins,double *planes,size_t pad);
  void (*grid_points)(const OpenMXForceGridPoints&,int first,int count,void *scratch);
  void (*nl_block)(const OpenMXForceProjectorSpecies&,const OpenMXProjectorForceBlock&,void *scratch);
};
extern "C" OPENMX_HOST_SIMD_EXPORT const ForceKernels* openmx_host_force_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(ForceKernels,force)
}
#endif
