#include "openmx_core.h"
#include <openmx/core/radial.hpp>
#include <openmx/core/bessel.hpp>
#include <openmx/core/fft_grid.hpp>
#include <climits>
#include <cmath>
#include <cstdio>
#include <exception>

struct OpenMXCoreContext {
  openmx::core::Context implementation;
};

extern "C" const char* openmx_core_version(void) { return "4.0 GPU"; }
extern "C" int openmx_core_fft_grid_supported(int size)
{
  return openmx::core::fft_grid_supported(size);
}

extern "C" int openmx_core_run(int argc, char** argv, OpenMXCoreEntry entry)
{
  if (!entry) return 1;
  try {
    OpenMXCoreContext context;
    return entry(argc, argv, &context);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "OpenMX core: %s\n", error.what());
  } catch (...) {
    std::fprintf(stderr, "OpenMX core: unexpected exception\n");
  }
  return 1;
}

extern "C" void openmx_core_finish(OpenMXCoreContext* context)
{
  if (context) context->implementation.finish();
}

extern "C" int openmx_core_has_accelerator(void)
{
  return openmx::core::has_accelerator();
}

extern "C" int openmx_core_spherical_bessel(double x, int lmax, double* sb, double* derivative)
{
  if (!sb || !derivative || !std::isfinite(x) || x < 0 || lmax < 0 || lmax > 29 ||
      x > (INT_MAX-lmax-11)/1.5) return 0;
  openmx::core::spherical_bessel(x,lmax,sb,derivative);
  return 1;
}

extern "C" int openmx_core_pao_transform(OpenMXCoreContext* context, int device,
    int lmax, const int* multiplicities, int radial_intervals, int k_count,
    double rmin, double radial_step, double k_step, void* user,
    OpenMXRadialSample sample, OpenMXRadialStore store,
    char* error, std::size_t error_capacity)
{
  if (error && error_capacity) error[0] = '\0';
  try {
    if (lmax < 0 || lmax > 29 || !multiplicities || !sample || !store)
      throw std::invalid_argument("invalid PAO transform interface arguments");
    auto samples = openmx::core::sample_radial(
      {lmax, radial_intervals, k_count, rmin, radial_step, k_step},
      {multiplicities, static_cast<std::size_t>(lmax+1)},
      [=](int l, int m, double radius) { return sample(user, l, m, radius); });
    auto result = context ? context->implementation.transform(samples, device)
                          : openmx::core::transform_host(samples);
    const auto values = result.view();
    for (std::size_t orbital = 0; orbital < samples.orbitals.size(); ++orbital)
      for (int k = 0; k < k_count; ++k)
        store(user, samples.orbitals[orbital].angular, samples.orbitals[orbital].multiplicity,
              k, openmx::core::element(values, orbital, k));
    return 1;
  } catch (const std::exception& exception) {
    if (error && error_capacity) std::snprintf(error, error_capacity, "%s", exception.what());
  } catch (...) {
    if (error && error_capacity) std::snprintf(error, error_capacity, "unexpected PAO transform exception");
  }
  return 0;
}
