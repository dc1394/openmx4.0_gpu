// exx-kernel-analytic: the four kernels in reciprocal space against 50-digit
// references (tests/fixtures/kernel_reference.txt, written by
// tools/exx_reference.py --kernels).
#include <openmx/exx/kernel.hpp>

#include "test_support.hpp"

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>

namespace exx_test {
namespace {
using namespace openmx::exx;

Kernel make_kernel(const std::string& name, double parameter)
{
  if (name == "overlap") return overlap_kernel();
  if (name == "coulomb") return coulomb_kernel();
  if (name == "short_range") return short_range_kernel(parameter);
  if (name == "truncated_coulomb") return truncated_coulomb_kernel(parameter);
  throw std::runtime_error("unknown kernel in the fixture: "+name);
}

template<class Call> bool throws_invalid(Call&& call)
{
  try { call(); } catch (const std::invalid_argument&) { return true; }
  return false;
}

} // namespace

void kernel_analytic(const std::string& fixture)
{
  std::ifstream in(fixture);
  require(in.good(),"cannot open the kernel fixture "+fixture);
  const double tolerance = 1e-13;
  Worst values("v~(kappa) against the 50-digit reference",tolerance);
  Worst weights("kappa^2 v~(kappa) against the 50-digit reference",tolerance);
  Worst limits("v~ at kappa = 0",tolerance);
  std::string line;
  bool header = false;
  long kinds[4] = {0,0,0,0};
  while (std::getline(in,line)) {
    std::istringstream words(line);
    std::string first, name;
    if (!(words >> first) || first[0] == '#') continue;
    if (first == "exx_kernel_fixture") {
      int version = 0;
      require(static_cast<bool>(words >> version) && version == 1,"unsupported kernel fixture version");
      header = true;
      continue;
    }
    require(header,"the kernel fixture has no header");
    double parameter = 0.0;
    require(static_cast<bool>(words >> name >> parameter),"malformed line in the kernel fixture: "+line);
    const Kernel kernel = make_kernel(name,parameter);
    if (first == "limit") {
      double reference = 0.0;
      require(static_cast<bool>(words >> reference),"malformed limit in the kernel fixture: "+line);
      limits.relative(kernel_limit(kernel),reference);
      limits.relative(kernel_value(kernel,0.0),reference);
    } else if (first == "point") {
      double kappa = 0.0, value = 0.0, weight = 0.0;
      require(static_cast<bool>(words >> kappa >> value >> weight),
              "malformed point in the kernel fixture: "+line);
      values.relative(kernel_value(kernel,kappa),value);
      weights.relative(kernel_weight(kernel,kappa),weight);
      ++kinds[static_cast<int>(kernel.kind)];
    } else {
      throw std::runtime_error("unknown record in the kernel fixture: "+first);
    }
  }
  for (long count : kinds) require(count >= 10,"the kernel fixture misses a kernel");
  values.finish();
  weights.finish();
  limits.finish();

  // The bare Coulomb kernel diverges at kappa = 0; its weight stays finite.
  const double infinity = std::numeric_limits<double>::infinity();
  require(kernel_limit(coulomb_kernel()) == infinity && kernel_value(coulomb_kernel(),0.0) == infinity,
          "the Coulomb kernel must diverge at kappa = 0");
  require(kernel_weight(coulomb_kernel(),0.0) == 4.0*std::numbers::pi,
          "kappa^2 v~ of the Coulomb kernel must be 4 pi at kappa = 0");
  require(kernel_weight(overlap_kernel(),0.0) == 0.0 &&
          kernel_weight(short_range_kernel(0.11),0.0) == 0.0 &&
          kernel_weight(truncated_coulomb_kernel(3.0),0.0) == 0.0,
          "kappa^2 v~ of the finite kernels must vanish at kappa = 0");

  // The short-range and the truncated kernel approach the bare Coulomb kernel
  // from below and never exceed it.
  for (double kappa : {0.05,0.4,3.0,25.0}) {
    const double bare = kernel_value(coulomb_kernel(),kappa);
    const double sr = kernel_value(short_range_kernel(0.11),kappa);
    const double tr = kernel_value(truncated_coulomb_kernel(3.0),kappa);
    require(sr > 0.0 && sr <= bare && tr >= 0.0 && tr <= 2.0*bare,"kernel bounds are violated");
  }
  require(relative_difference(kernel_value(short_range_kernel(0.11),60.0),
                              kernel_value(coulomb_kernel(),60.0)) < 1e-15,
          "the short-range kernel must equal the Coulomb kernel at large kappa");

  // Invalid arguments are rejected.
  require(throws_invalid([] { (void)short_range_kernel(0.0); }) &&
          throws_invalid([] { (void)short_range_kernel(-0.11); }) &&
          throws_invalid([] { (void)truncated_coulomb_kernel(std::numeric_limits<double>::quiet_NaN()); }) &&
          throws_invalid([] { (void)kernel_value(overlap_kernel(),-1.0); }) &&
          throws_invalid([] { (void)kernel_weight(coulomb_kernel(),std::numeric_limits<double>::infinity()); }) &&
          throws_invalid([] { (void)kernel_value(Kernel{KernelKind::short_range,0.0},1.0); }) &&
          throws_invalid([] { (void)kernel_limit(Kernel{static_cast<KernelKind>(17),0.0}); }),
          "invalid kernel arguments must throw std::invalid_argument");
}

} // namespace exx_test
