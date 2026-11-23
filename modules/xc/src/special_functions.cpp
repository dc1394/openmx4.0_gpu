#include "special_functions.hpp"

#include <openmx/xc/functionals.hpp>

#include <cmath>
#include <limits>

// exp(x) E1(x) and erfcx(x) in double precision.  Small arguments use the
// power series of E1 and the product exp(x^2) erfc(x); larger arguments use
// the continued fractions
//
//   exp(x) E1(x) = 1 / (x + 1 - 1 / (x + 3 - 4 / (x + 5 - 9 / (x + 7 - ...))))
//   erfcx(x)     = (1/sqrt(pi)) / (x + (1/2) / (x + 1 / (x + (3/2) / (x + ...))))
//
// evaluated from a fixed depth upwards.  The depths were chosen so that the
// truncation is below the rounding error on each interval, with a margin of
// one step of the table.  The derivatives are taken from the same fraction
// (exp(x) E1(x) - 1/x and 2 x erfcx(x) - 2/sqrt(pi) cancel for large x).
// tests/test_wpbeh.cpp compares values and derivatives with 50-digit
// references between 1e-30 and 1e8.

namespace openmx::xc::detail {
namespace {

constexpr double euler_gamma = 0.57721566490153286061;
constexpr double two_over_sqrt_pi = 1.1283791670955125739;
constexpr double one_over_sqrt_pi = 0.56418958354775628695;

int e1_fraction_depth(double x) noexcept
{
  if (x < 1.5) return 120;
  if (x < 2.0) return 100;
  if (x < 3.0) return 60;
  if (x < 5.0) return 48;
  if (x < 10.0) return 32;
  if (x < 20.0) return 20;
  return 16;
}

int erfcx_fraction_depth(double x) noexcept
{
  if (x < 2.5) return 100;
  if (x < 3.0) return 80;
  if (x < 4.0) return 50;
  if (x < 6.0) return 30;
  if (x < 10.0) return 20;
  return 12;
}

}  // namespace

ValueAndDerivative e1_scaled(double x) noexcept
{
  if (!(x > 0.0)) {
    if (x == 0.0) return {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan};
  }
  if (x < 1.0) {
    // E1(x) = -gamma - ln x - sum_{k >= 1} (-x)^k / (k k!)
    double sum = 0.0, term = 1.0;
    for (int k = 1; k <= 40; ++k) {
      term *= -x / k;
      const double add = -term / k;
      sum += add;
      if (std::fabs(add) <= 1e-17 * std::fabs(sum)) break;
    }
    const double value = std::exp(x) * (-euler_gamma - std::log(x) + sum);
    return {value, value - 1.0 / x};
  }
  double tail = 0.0;
  for (int k = e1_fraction_depth(x); k >= 1; --k) tail = (static_cast<double>(k) * k) / (x + (2 * k + 1) - tail);
  const double value = 1.0 / (x + 1.0 - tail);
  return {value, (tail - 1.0) * value / x};
}

ValueAndDerivative erfcx(double x) noexcept
{
  if (std::isnan(x)) return {x, x};
  if (x < 0.0) {
    const double value = 2.0 * std::exp(x * x) - erfcx(-x).value;
    return {value, 2.0 * x * value - two_over_sqrt_pi};
  }
  if (x < 2.0) {
    const double value = std::exp(x * x) * std::erfc(x);
    return {value, 2.0 * x * value - two_over_sqrt_pi};
  }
  double tail = 0.0;
  for (int k = erfcx_fraction_depth(x); k >= 1; --k) tail = (0.5 * k) / (x + tail);
  return {one_over_sqrt_pi / (x + tail), -two_over_sqrt_pi * tail / (x + tail)};
}

}  // namespace openmx::xc::detail

namespace openmx::xc {

double e1_scaled(double x) noexcept { return detail::e1_scaled(x).value; }

double erfcx(double x) noexcept { return detail::erfcx(x).value; }

}  // namespace openmx::xc
