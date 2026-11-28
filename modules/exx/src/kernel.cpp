#include <openmx/exx/kernel.hpp>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace openmx::exx {
namespace {

constexpr double four_pi = 4.0*std::numbers::pi;

double checked_parameter(double value, const char* message)
{
  if (!std::isfinite(value) || value <= 0.0) throw std::invalid_argument(message);
  return value;
}

void check(const Kernel& kernel, double kappa)
{
  if (!std::isfinite(kappa) || kappa < 0.0)
    throw std::invalid_argument("the reciprocal coordinate of a kernel must be finite and nonnegative");
  switch (kernel.kind) {
    case KernelKind::overlap:
    case KernelKind::coulomb:
      return;
    case KernelKind::short_range:
      checked_parameter(kernel.parameter,"the screening parameter must be finite and positive");
      return;
    case KernelKind::truncated_coulomb:
    case KernelKind::truncation_derivative:
      checked_parameter(kernel.parameter,"the truncation radius must be finite and positive");
      return;
  }
  throw std::invalid_argument("unknown kernel");
}

// (1 - exp(-t))/t for t >= 0, with the limit 1 at t = 0.
double one_minus_exp_over(double t)
{
  return t > 0.0 ? -std::expm1(-t)/t : 1.0;
}

// (sin(y)/y)^2 for y >= 0, with the limit 1 at y = 0.
double sinc_squared(double y)
{
  if (y <= 0.0) return 1.0;
  const double s = std::sin(y)/y;
  return s*s;
}

} // namespace

Kernel overlap_kernel() noexcept { return {KernelKind::overlap,0.0}; }
Kernel coulomb_kernel() noexcept { return {KernelKind::coulomb,0.0}; }

Kernel short_range_kernel(double omega)
{
  return {KernelKind::short_range,
          checked_parameter(omega,"the screening parameter must be finite and positive")};
}

Kernel truncated_coulomb_kernel(double cutoff_radius)
{
  return {KernelKind::truncated_coulomb,
          checked_parameter(cutoff_radius,"the truncation radius must be finite and positive")};
}

Kernel truncation_derivative_kernel(double cutoff_radius)
{
  return {KernelKind::truncation_derivative,
          checked_parameter(cutoff_radius,"the truncation radius must be finite and positive")};
}

double kernel_limit(const Kernel& kernel)
{
  check(kernel,0.0);
  switch (kernel.kind) {
    case KernelKind::overlap: return 1.0;
    case KernelKind::coulomb: return std::numeric_limits<double>::infinity();
    case KernelKind::short_range: return std::numbers::pi/(kernel.parameter*kernel.parameter);
    case KernelKind::truncated_coulomb: return 2.0*std::numbers::pi*kernel.parameter*kernel.parameter;
    case KernelKind::truncation_derivative: return four_pi*kernel.parameter;
  }
  throw std::invalid_argument("unknown kernel");
}

double kernel_value(const Kernel& kernel, double kappa)
{
  check(kernel,kappa);
  switch (kernel.kind) {
    case KernelKind::overlap:
      return 1.0;
    case KernelKind::coulomb:
      return kappa > 0.0 ? four_pi/(kappa*kappa) : std::numeric_limits<double>::infinity();
    case KernelKind::short_range: {
      // 4 pi (1 - exp(-t))/kappa^2 with t = (kappa/(2 omega))^2 is (pi/omega^2) (1 - exp(-t))/t.
      const double x = 0.5*kappa/kernel.parameter;
      return std::numbers::pi/(kernel.parameter*kernel.parameter)*one_minus_exp_over(x*x);
    }
    case KernelKind::truncated_coulomb: {
      // 1 - cos(2 y) = 2 sin(y)^2 with y = kappa R_c / 2.
      const double y = 0.5*kappa*kernel.parameter;
      return 2.0*std::numbers::pi*kernel.parameter*kernel.parameter*sinc_squared(y);
    }
    case KernelKind::truncation_derivative:
      return kappa > 0.0 ? four_pi*std::sin(kappa*kernel.parameter)/kappa : four_pi*kernel.parameter;
  }
  throw std::invalid_argument("unknown kernel");
}

double kernel_weight(const Kernel& kernel, double kappa)
{
  check(kernel,kappa);
  switch (kernel.kind) {
    case KernelKind::overlap:
      return kappa*kappa;
    case KernelKind::coulomb:
      return four_pi;
    case KernelKind::short_range: {
      const double x = 0.5*kappa/kernel.parameter;
      return -four_pi*std::expm1(-x*x);
    }
    case KernelKind::truncated_coulomb: {
      const double s = std::sin(0.5*kappa*kernel.parameter);
      return 2.0*four_pi*s*s;
    }
    case KernelKind::truncation_derivative:
      return four_pi*kappa*std::sin(kappa*kernel.parameter);
  }
  throw std::invalid_argument("unknown kernel");
}

} // namespace openmx::exx
