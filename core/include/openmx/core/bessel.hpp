#pragma once

#include <Kokkos_Core.hpp>
#include <openmx/basis/detail/bessel.hpp>

namespace openmx::core {
namespace detail {
struct BesselMath {
  KOKKOS_INLINE_FUNCTION static double sin(double x) { return Kokkos::sin(x); }
  KOKKOS_INLINE_FUNCTION static double cos(double x) { return Kokkos::cos(x); }
  KOKKOS_INLINE_FUNCTION static double abs(double x) { return Kokkos::abs(x); }
};
} // namespace detail

// Preserve the unchecked device-callable boundary and its math provider.
// 0 <= lmax <= 29; finite nonnegative x with lmax+int(1.5*x)+10 in int.
// Values need lmax+2 elements, optional derivatives need lmax+1.
KOKKOS_INLINE_FUNCTION
void spherical_bessel(double x, int lmax, double* sb, double* derivative)
{
  openmx::basis::detail::spherical_bessel<detail::BesselMath>(x,lmax,sb,derivative);
}

// Values only: lmax+1 elements; retains the large-argument fast path.
KOKKOS_INLINE_FUNCTION
void spherical_bessel_values(double x, int lmax, double* sb)
{
  openmx::basis::detail::spherical_bessel_values<detail::BesselMath>(x,lmax,sb);
}
} // namespace openmx::core
