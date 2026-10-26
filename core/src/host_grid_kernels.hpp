#ifndef OPENMX_HOST_GRID_KERNELS_HPP
#define OPENMX_HOST_GRID_KERNELS_HPP
#include "openmx_core_grid.h"

// Private, ordinary C++ boundary. Neither Kokkos vector types nor ISA flags
// cross into the C ABI, device translation units or ownership layer.
namespace openmx::core::host {
struct GridScratch {
  double *dm, *orb0, *orb1, *h;
  size_t orbitals;
};
struct GridKernels {
  int width;
  void (*density)(const OpenMXGridPair&, GridScratch, double* const*, int);
  void (*hamiltonian)(const OpenMXGridPair&, GridScratch, int);
  void (*packed_hamiltonian)(const OpenMXGridPackedHamiltonian&);
};
#if defined(_WIN32)
#ifdef OPENMX_HOST_SIMD_BUILD
#define OPENMX_HOST_EXPORT __declspec(dllexport)
#else
#define OPENMX_HOST_EXPORT __declspec(dllimport)
#endif
#else
#define OPENMX_HOST_EXPORT __attribute__((visibility("default")))
#endif
extern "C" OPENMX_HOST_EXPORT const GridKernels* openmx_host_grid_kernels(int mode) noexcept;
extern "C" OPENMX_HOST_EXPORT int openmx_host_grid_mode(const char* requested) noexcept;
#undef OPENMX_HOST_EXPORT
const GridKernels& grid_scalar() noexcept;
const GridKernels& grid_avx2() noexcept;
const GridKernels& grid_avx512() noexcept;
const GridKernels& grid_neon() noexcept;
// The committed kernels (host_grid_committed.cpp, one per ISA), used by the
// grid kernels of the same ISA for the shapes where they are faster.
const GridKernels& grid_committed_scalar() noexcept;
const GridKernels& grid_committed_avx2() noexcept;
const GridKernels& grid_committed_avx512() noexcept;
const GridKernels& grid_committed_neon() noexcept;
}
#endif
