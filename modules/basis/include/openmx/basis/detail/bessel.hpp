#pragma once

// This numerical implementation is shared by the independent host library
// and the application's Kokkos adapters. Math supplies sin/cos/abs; execution
// annotations do not introduce a dependency on any GPU SDK or runtime.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define OPENMX_BASIS_BESSEL_INLINE __host__ __device__ inline
#else
#define OPENMX_BASIS_BESSEL_INLINE inline
#endif

namespace openmx::basis::detail {

/* Downward recurrence used by OpenMX's Spherical_Bessel (T. Ozaki, 2005).
   Preconditions: 0 <= lmax <= 29; x is finite, nonnegative, and small
   enough that lmax + int(1.5*x) + 10 fits in int. sb has lmax+2 entries.
   This pure kernel has identical code on host, CUDA, and HIP. */
template<class Math>
OPENMX_BASIS_BESSEL_INLINE
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
  const double sine = Math::sin(x), cosine = Math::cos(x);
  const double j0 = sine*invx, j1 = sine*invx*invx - cosine*invx;
  const double scale = Math::abs(tmp[1]) < Math::abs(tmp[0]) ? j0/tmp[0] : j1/tmp[1];
  for (int l = 0; l <= lmax+1; ++l) sb[l] = tmp[l]*scale;
  if (derivative) {
    derivative[0] = cosine*invx - sine*invx*invx;
    for (int l = 1; l <= lmax; ++l)
      derivative[l] = (l*sb[l-1] - (l+1.0)*sb[l+1])/(2.0*l+1.0);
  }
}

/* Values-only: same input preconditions as above; sb needs lmax+1 entries.
   Retain downward recurrence for small arguments and near the transition.
   Well above the largest order, upward recurrence is stable and takes only
   O(lmax) steps instead of O(x). The factor two is a conservative margin.
   See DLMF 10.49 (j0,j1), 10.51.1 and 10.74(iv).
   This does not replace the legacy values-and-derivatives kernel above. */
template<class Math>
OPENMX_BASIS_BESSEL_INLINE
void spherical_bessel_values(double x, int lmax, double* sb)
{
  if (lmax == 0) {
    sb[0] = x == 0.0 ? 1.0 : Math::sin(x)/x;
  } else if (x > 2.0*(lmax+1)) {
    const double invx = 1.0/x;
    const double sine = Math::sin(x), cosine = Math::cos(x);
    sb[0] = sine*invx;
    sb[1] = sine*invx*invx - cosine*invx;
    for (int l = 1; l < lmax; ++l)
      sb[l+1] = (2.0*l+1.0)*invx*sb[l] - sb[l-1];
  } else {
    // The legacy recurrence also writes order lmax+1. Keep its scratch
    // private so the values-only interface never writes past its contract.
    double work[32];
    spherical_bessel<Math>(x,lmax,work,nullptr);
    for (int l = 0; l <= lmax; ++l) sb[l] = work[l];
  }
}

} // namespace openmx::basis::detail

#undef OPENMX_BASIS_BESSEL_INLINE
