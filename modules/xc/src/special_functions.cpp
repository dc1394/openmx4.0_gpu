// exp(x) E1(x) and erfcx(x) in double precision.  The arithmetic, with the
// description of the series and the continued fractions, is in
// openmx/xc/hybrid_kernels.h, shared with C callers and with the
// application's GPU device code.  tests/test_wpbeh.cpp compares values and
// derivatives with 50-digit references between 1e-30 and 1e8.

#include "special_functions.hpp"

#include <openmx/xc/functionals.hpp>
#include <openmx/xc/hybrid_kernels.h>

namespace openmx::xc::detail {

ValueAndDerivative e1_scaled(double x) noexcept
{
  double out[2];
  openmx_xc_kernel_e1_scaled(x, out);
  return {out[0], out[1]};
}

ValueAndDerivative erfcx(double x) noexcept
{
  double out[2];
  openmx_xc_kernel_erfcx(x, out);
  return {out[0], out[1]};
}

}  // namespace openmx::xc::detail

namespace openmx::xc {

double e1_scaled(double x) noexcept { return detail::e1_scaled(x).value; }

double erfcx(double x) noexcept { return detail::erfcx(x).value; }

}  // namespace openmx::xc
