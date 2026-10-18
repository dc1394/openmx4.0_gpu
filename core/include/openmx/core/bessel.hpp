#pragma once

#include <Kokkos_Core.hpp>

namespace openmx::core {

/* Downward recurrence used by OpenMX's Spherical_Bessel (T. Ozaki, 2005).
   Preconditions: 0 <= lmax <= 29; x is finite, nonnegative, and small
   enough that lmax + int(1.5*x) + 10 fits in int. sb has lmax+2 entries.
   This pure kernel has identical code on host, CUDA, and HIP. */
KOKKOS_INLINE_FUNCTION
void spherical_bessel(double x, int lmax, double* sb, double* derivative)
{
  if (x == 0.0) {
    for (int l = 0; l <= lmax+1; ++l) sb[l] = 0.0;
    sb[0] = 1.0;
    if (derivative) {
      for (int l = 0; l <= lmax; ++l) derivative[l] = l == 1 ? 1.0/3.0 : 0.0;
    }
    return;
  }
  int nmax = lmax + static_cast<int>(1.5*x) + 10;
  if (nmax < 30) nmax = 30;
  const double invx = 1.0/x;
  double previous = 0.0, current = 1.0e-14, older = 0.0;
  for (int n = nmax-1; lmax+2 < n; --n) {
    double next = (2.0*n + 1.0)*invx*current - previous;
    if (0.0 < next-1.0e250) {
      current *= 1.0/next;
      next = 1.0;
    }
    older = previous;
    previous = current;
    current = next;
  }
  double tmp[40];
  const int nstart = lmax+3;
  tmp[nstart-1] = current;
  tmp[nstart] = previous;
  tmp[nstart+1] = older;
  const double divisor = tmp[nstart-1];
  tmp[nstart-1] /= divisor;
  tmp[nstart] /= divisor;
  for (int n = lmax+2; n > 0; --n) {
    tmp[n-1] = (2.0*n+1.0)*invx*tmp[n] - tmp[n+1];
    if (1.0e250 < tmp[n-1]) {
      const double scale = tmp[n-1];
      for (int m = n-1; m <= lmax+1; ++m) tmp[m] /= scale;
    }
  }
  const double sine = Kokkos::sin(x), cosine = Kokkos::cos(x);
  const double j0 = sine*invx, j1 = sine*invx*invx - cosine*invx;
  const double scale = Kokkos::abs(tmp[1]) < Kokkos::abs(tmp[0]) ? j0/tmp[0] : j1/tmp[1];
  for (int l = 0; l <= lmax+1; ++l) sb[l] = tmp[l]*scale;
  if (derivative) {
    derivative[0] = cosine*invx - sine*invx*invx;
    for (int l = 1; l <= lmax; ++l)
      derivative[l] = (l*sb[l-1] - (l+1.0)*sb[l+1])/(2.0*l+1.0);
  }
}

} // namespace openmx::core
