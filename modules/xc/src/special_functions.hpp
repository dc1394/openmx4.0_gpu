#ifndef OPENMX_XC_SRC_SPECIAL_FUNCTIONS_HPP
#define OPENMX_XC_SRC_SPECIAL_FUNCTIONS_HPP

// Internal to the xc library: the two special functions of the short-range
// exchange together with their first derivatives.  The public functions
// e1_scaled() and erfcx() of functionals.hpp return the values.

namespace openmx::xc::detail {

struct ValueAndDerivative {
  double value, derivative;
};

// exp(x) E1(x) and its derivative exp(x) E1(x) - 1/x, for x > 0.
// x = 0 gives (+infinity, -infinity); negative x and NaN give NaN.
ValueAndDerivative e1_scaled(double x) noexcept;

// erfcx(x) = exp(x^2) erfc(x) and its derivative 2 x erfcx(x) - 2/sqrt(pi).
ValueAndDerivative erfcx(double x) noexcept;

}  // namespace openmx::xc::detail

#endif
