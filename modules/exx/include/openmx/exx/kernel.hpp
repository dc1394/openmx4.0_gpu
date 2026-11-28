#pragma once

namespace openmx::exx {

// Interaction kernels of the exact exchange in reciprocal space.  v(r) is the
// kernel between two points at distance r (bohr) and
//   v~(kappa) = int v(r) exp(-i kappa.r) d^3r
// its Fourier transform (kappa in bohr^-1).  All values are in Hartree atomic
// units.
//
//   overlap            v = delta(r)               v~ = 1
//   coulomb            v = 1/r                    v~ = 4 pi / kappa^2
//   short_range        v = erfc(omega r)/r        v~ = 4 pi (1 - exp(-kappa^2/(4 omega^2))) / kappa^2
//   truncated_coulomb  v = theta(R_c - r)/r       v~ = 4 pi (1 - cos(kappa R_c)) / kappa^2
enum class KernelKind { overlap, coulomb, short_range, truncated_coulomb, truncation_derivative };

// A kernel and its single parameter: omega (bohr^-1, positive) for
// short_range, R_c (bohr, positive) for truncated_coulomb and its derivative. The other two
// kernels ignore the parameter.  Use the factory functions below; they check
// the parameter and throw std::invalid_argument for a value that is not
// finite and positive.
struct Kernel {
  KernelKind kind = KernelKind::overlap;
  double parameter = 0.0;
};

Kernel overlap_kernel() noexcept;
Kernel coulomb_kernel() noexcept;
Kernel short_range_kernel(double omega);
Kernel truncated_coulomb_kernel(double cutoff_radius);
// Derivative of the truncated Coulomb kernel with respect to R_c.
Kernel truncation_derivative_kernel(double cutoff_radius);

// v~(kappa) for a finite kappa >= 0.  At kappa = 0 the function returns the
// limit of the table in kernel_limit(); for the bare Coulomb kernel that is
// +infinity.  The short-range and the truncated kernel are evaluated without
// cancellation for small kappa (expm1 and sin of the half argument).
// Throws std::invalid_argument for a negative or non-finite kappa and for an
// invalid kernel.
double kernel_value(const Kernel& kernel, double kappa);

// kappa^2 v~(kappa), the factor that enters the radial integrals.  It is
// finite for every kernel at kappa = 0 (4 pi for the bare Coulomb kernel,
// 0 for the other kernels).
double kernel_weight(const Kernel& kernel, double kappa);

// lim_{kappa -> 0} v~(kappa): 1, +infinity, pi/omega^2, 2 pi R_c^2 and 4 pi R_c.
double kernel_limit(const Kernel& kernel);

} // namespace openmx::exx
