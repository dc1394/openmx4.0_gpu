#include <openmx/core/pair_bessel.hpp>
#include "openmx_core_pair_bessel.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using namespace openmx::core;

void require(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance = 2e-12)
{
  require(std::isfinite(actual) && std::abs(actual-expected) <= tolerance*(1+std::abs(expected)),
          "pair Bessel numerical mismatch");
}

template<class Function> void rejects(Function&& function)
{
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  require(rejected,"invalid pair Bessel input accepted");
}

// Independent power series for this test's arguments 0 <= x <= 6. libc++
// does not provide std::sph_bessel. Production uses downward recurrence;
// the reference deliberately uses a different algorithm on every platform.
double reference_bessel(int order, double argument)
{
  long double term = 1;
  for (int j = 1; j <= order; ++j) term *= argument/(2*j+1.L);
  long double sum = term;
  for (int j = 1; j < 100; ++j) {
    term *= -static_cast<long double>(argument)*argument/(2*j*(2*order+2*j+1.L));
    sum += term;
    if (std::abs(term) <= std::numeric_limits<long double>::epsilon()*std::abs(sum))
      return static_cast<double>(sum);
  }
  throw std::runtime_error("pair Bessel reference series did not converge");
}

void values_test()
{
  const std::array wavevectors{0.,.125,.5,1.,3.,8.};
  PairBesselWorkspace workspace(29,wavevectors,4);
  for (int angular : {0,1,4,12,29}) {
    const auto table = workspace.sample(.75,angular);
    for (std::size_t point = 0; point < wavevectors.size(); ++point) {
      const double argument = wavevectors[point]*.75;
      for (int order = 0; order <= angular; ++order) {
        const double expected = argument == 0 ? (order == 0 ? 1. : 0.)
                                             : reference_bessel(order,argument);
        const double derivative = argument == 0 ? (order == 1 ? 1./3 : 0.)
          : order*expected/argument-reference_bessel(order+1,argument);
        near(element(table.values,0,order,point),expected);
        near(element(table.values,1,order,point),derivative);
      }
    }
  }
}

void cache_test()
{
  std::array wavevectors{0.,.5,1.,2.};
  PairBesselWorkspace workspace(4,wavevectors,2);
  const auto first = workspace.sample(1.,2);
  const double* first_address = first.values.data_handle();
  wavevectors[2] = 25;
  const auto second = workspace.sample(2.,1);
  near(element(first.values,0,0,2),std::sin(1.));
  near(element(second.values,0,0,2),std::sin(2.)/2);
  const auto repeated = workspace.sample(1.,2);
  require(repeated.values.data_handle() == first_address,"cache hit copied the table");
  const auto overflow = workspace.sample(3.,4);
  const double* scratch_address = overflow.values.data_handle();
  const auto another = workspace.sample(4.,0);
  require(another.values.data_handle() == scratch_address,"full cache reallocates scratch");
  near(element(first.values,0,0,2),std::sin(1.));
  require(workspace.cached_tables() == 2,"pair Bessel cache limit exceeded");
  require(workspace.statistics().hits == 1 && workspace.statistics().misses == 4,
          "pair Bessel cache statistics mismatch");
  PairBesselWorkspace disabled(4,{wavevectors.data(),1},0);
  const auto uncached = disabled.sample(0.,2);
  const auto signed_zero = disabled.sample(-0.,2);
  require(uncached.values.data_handle() == signed_zero.values.data_handle(),"disabled scratch changed");
  require(disabled.cached_tables() == 0 && disabled.statistics().hits == 0,"disabled cache retains entries");
  PairBesselWorkspace zeros(4,{wavevectors.data(),1},2);
  const auto zero = zeros.sample(0.,2);
  const auto negative_zero = zeros.sample(-0.,2);
  require(zero.values.data_handle() == negative_zero.values.data_handle(),"signed-zero cache equality changed");
  const auto different_order = zeros.sample(0.,1);
  require(different_order.values.data_handle() != zero.values.data_handle(),"angular order omitted from key");
  PairBesselWorkspace independent(4,{wavevectors.data(),1},2);
  require(independent.sample(0.,2).values.data_handle() != zero.values.data_handle(),
          "independent workspaces share storage");
}

void failure_test()
{
  const std::array wavevectors{0.,1.,2.};
  rejects([&] { PairBesselWorkspace invalid(-1,wavevectors,1); });
  rejects([&] { PairBesselWorkspace invalid(30,wavevectors,1); });
  rejects([&] { PairBesselWorkspace invalid(2,{},1); });
  for (double invalid : {-1.,std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    const std::array bad_grid{0.,invalid};
    rejects([&] { PairBesselWorkspace invalid_grid(2,bad_grid,1); });
  }
  PairBesselWorkspace workspace(4,wavevectors,2);
  const auto retained = workspace.sample(1.,2);
  for (double invalid : {-1.,std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(),std::numeric_limits<double>::max()})
    rejects([&] { (void)workspace.sample(invalid,2); });
  rejects([&] { (void)workspace.sample(1.,-1); });
  rejects([&] { (void)workspace.sample(1.,5); });
  near(element(retained.values,0,0,1),std::sin(1.));
  require(workspace.statistics().misses == 1,"rejected sample modified cache");
}

void c_api_test()
{
  const std::array wavevectors{0.,1.,2.};
  char error[128];
  const auto destroy = &openmx_core_pair_bessel_destroy;
  std::unique_ptr<OpenMXPairBesselWorkspace,decltype(destroy)> workspace(
    openmx_core_pair_bessel_create(4,3,wavevectors.data(),2,error,sizeof(error)),destroy);
  require(workspace != nullptr,error);
  OpenMXPairBesselTable table{};
  require(openmx_core_pair_bessel_sample(workspace.get(),1.,2,&table,error,sizeof(error)),error);
  require(table.points == 3 && table.angular_rows == 3,"C table dimensions mismatch");
  near(table.values[1],std::sin(1.));
  near(table.derivatives[1],std::cos(1.)-std::sin(1.));
  const auto previous = table;
  require(!openmx_core_pair_bessel_sample(workspace.get(),-1.,2,&table,error,sizeof(error)),
          "C sample accepted negative distance");
  require(error[0] && table.values == previous.values && table.derivatives == previous.derivatives,
          "C sample failure overwrote result");
  require(!openmx_core_pair_bessel_sample(nullptr,1.,2,&table,error,sizeof(error)),
          "C sample accepted null workspace");
  require(!openmx_core_pair_bessel_sample(workspace.get(),1.,2,nullptr,error,sizeof(error)),
          "C sample accepted null output");
  require(!openmx_core_pair_bessel_create(2,0,wavevectors.data(),0,error,sizeof(error)),
          "C create accepted empty grid");
  require(!openmx_core_pair_bessel_create(2,3,nullptr,0,error,sizeof(error)),
          "C create accepted null grid");
  long hits = -1, misses = -1;
  openmx_core_pair_bessel_statistics(workspace.get(),&hits,&misses);
  require(hits == 0 && misses == 1,"C statistics mismatch");
  openmx_core_pair_bessel_statistics(nullptr,&hits,&misses);
  require(hits == 0 && misses == 0,"null statistics mismatch");
  openmx_core_pair_bessel_statistics(workspace.get(),nullptr,nullptr);
  openmx_core_pair_bessel_destroy(nullptr);
}

} // namespace

int main(int argc, char** argv)
{
  try {
    const std::string name = argc == 2 ? argv[1] : "";
    if (name == "values") values_test();
    else if (name == "cache") cache_test();
    else if (name == "failure") failure_test();
    else if (name == "c-api") c_api_test();
    else throw std::runtime_error("unknown pair Bessel test");
    std::cout << "PASS pair Bessel " << name << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
