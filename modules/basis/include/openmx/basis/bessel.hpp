#pragma once

#include <array>
#include <span>

namespace openmx::basis {

// Spherical Bessel j_l(x), with derivatives with respect to dimensionless x
// (for x=k*r, a radial derivative additionally needs the factor k).
// Owns its results without heap allocation. Values cover 0..maximum_order+1;
// derivatives cover 0..maximum_order. Borrowed spans expire with this object.
// Uses OpenMX's downward recurrence and its original rounding, including the
// subtraction used for j_0'. At x=0 it returns the analytic limiting values.
// Orders 0..29 and finite nonnegative arguments within the recurrence's int
// range are accepted. Cost is O(maximum_order+x). Invalid inputs throw before
// evaluation. The inherited recurrence can underflow/overflow for extreme
// floating-point inputs.
class SphericalBesselSample {
 public:
  SphericalBesselSample(double argument, int maximum_order);
  int maximum_order() const noexcept { return maximum_order_; }
  std::span<const double> values() const noexcept {
    return {values_.data(), static_cast<std::size_t>(maximum_order_+2)};
  }
  std::span<const double> derivatives() const noexcept {
    return {derivatives_.data(), static_cast<std::size_t>(maximum_order_+1)};
  }
 private:
  int maximum_order_;
  std::array<double,31> values_{};
  std::array<double,30> derivatives_{};
};

} // namespace openmx::basis
