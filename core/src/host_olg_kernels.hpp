#ifndef OPENMX_HOST_OLG_KERNELS_HPP
#define OPENMX_HOST_OLG_KERNELS_HPP
#include "openmx_core_olg.h"
#include "host_simd_dispatch.hpp"
#include <cstddef>

// Private, ordinary C++ boundary of the overlap-grid family. No Kokkos
// vector type or ISA flag crosses into the C ABI or device translation units.
namespace openmx::core::host {
struct OlgKernels {
  int width;
  void (*intersect)(const OpenMXOlgLists&,const OpenMXOlgQuery*,size_t,int*);
  void (*intersect_lists)(const OpenMXOlgLists&,const OpenMXOlgNeighbours&,size_t,int*);
};
extern "C" OPENMX_HOST_SIMD_EXPORT const OlgKernels* openmx_host_olg_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OlgKernels,olg)
}
#endif
