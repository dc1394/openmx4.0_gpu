#pragma once
#include "host_simd_dispatch.hpp"
#include <cstddef>

namespace openmx::core {
enum class RadialWeighting;
struct ProductChannel;

// POD arguments shared by Core and its separately compiled host SIMD library.
struct HostRadialProblem {
  int lmax, radial_count, k_count;
  std::size_t channel_count;
  RadialWeighting weighting;
  double step;
  const double *radii, *measures, *wavevectors, *radial_values;
  const ProductChannel* channels;
  double* result;
};

namespace host {
struct RadialKernels {
  int width;
  void (*transform)(const HostRadialProblem&);
};
extern "C" OPENMX_HOST_SIMD_EXPORT const RadialKernels* openmx_host_radial_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(RadialKernels,radial)
}
}
