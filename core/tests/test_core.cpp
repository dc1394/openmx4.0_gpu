#include <openmx/core/radial.hpp>
#include <openmx/core/bessel.hpp>
#include <openmx/core/fft_grid.hpp>
#include "openmx_periodic.h"
#include "openmx_core.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>
#include <type_traits>

namespace {
using namespace openmx::core;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance = 3e-12) {
  if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance*(1+std::abs(expected))) {
    std::ostringstream message;
    message << std::setprecision(17) << "numerical mismatch: " << actual << " vs " << expected;
    throw std::runtime_error(message.str());
  }
}
template<class F> void rejects(F&& operation) {
  bool rejected = false;
  try { operation(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "invalid input was accepted");
}

double reference_bessel(int l, double x) {
#if defined(__GLIBCXX__)
  return std::sph_bessel(l,x);
#else
  // libc++ does not provide the optional special functions. Use independent
  // series/closed forms, and a Legendre integral for higher orders.
  if (x < 1) {
    long double term = 1;
    for (int j = 1; j <= l; ++j) term *= x/(2*j+1);
    long double sum = term;
    for (int j = 1; j < 40; ++j) {
      term *= -static_cast<long double>(x)*x/(2*j*(2*l+2*j+1));
      sum += term;
    }
    return static_cast<double>(sum);
  }
  if (l <= 3) {
    long double a = std::sin(static_cast<long double>(x))/x;
    long double b = a/x-std::cos(static_cast<long double>(x))/x;
    if (l == 0) return static_cast<double>(a);
    for (int j = 1; j < l; ++j) {
      long double next = (2*j+1)*b/x-a;
      a = b; b = next;
    }
    return static_cast<double>(b);
  }
  constexpr int steps = 32768;
  long double sum = 0;
  for (int i = 0; i <= steps; ++i) {
    const long double t = static_cast<long double>(i)/steps;
    long double a = 1, b = t;
    for (int j = 1; j < l; ++j) {
      const long double next = ((2*j+1)*t*b-j*a)/(j+1);
      a = b; b = next;
    }
    const long double wave = l%2 ? std::sin(x*t) : std::cos(x*t);
    sum += (i == 0 || i == steps ? 1 : i%2 ? 4 : 2)*wave*b;
  }
  return static_cast<double>((l%4 < 2 ? 1 : -1)*sum/(3*steps));
#endif
}

void tensor_test() {
  Tensor<double,3> first({2,3,4});
  auto view = first.view();
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) element(view,i,j,k) = 100*i+10*j+k;
  require(first.data()[23] == 123, "tensor layout is not row major");
  auto second = first;
  element(second.view(),0,0,0) = -7;
  require(element(first.view(),0,0,0) == 0, "tensor copies share ownership");
  Tensor<double,3> assigned({1,1,1});
  assigned = first;
  require(assigned.extent(2) == 4 && assigned.data()[23] == 123,
          "copy assignment lost shape or data");
  struct ThrowOnCopy {
    bool fail = false;
    ThrowOnCopy() = default;
    ThrowOnCopy(const ThrowOnCopy& other) : fail(other.fail) {
      if (fail) throw std::runtime_error("injected element copy failure");
    }
    ThrowOnCopy& operator=(const ThrowOnCopy&) = default;
  };
  Tensor<ThrowOnCopy,1> source({2}), target({3});
  source.data()[0].fail = true;
  rejects([&] { target = source; });
  require(target.extent(0) == 3 && target.size() == 3 && !target.data()[0].fail,
          "failed copy assignment corrupted tensor ownership");
  auto moved = std::move(first);
  require(moved.data()[23] == 123 && first.extent(0) == 0, "invalid moved tensor");
  second = std::move(moved);
  require(second.data()[23] == 123 && moved.extent(1) == 0, "invalid move assignment");
  const auto& constant = second;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(element(constant.view(),0,0,0))>>);
  near(element(constant.view(),1,2,3),123);
  require(Tensor<int,2>({0,19}).size() == 0, "empty tensor allocates entries");
  rejects([] { Tensor<double,2> huge({std::numeric_limits<std::size_t>::max(),2}); });
  rejects([&] { (void)second.extent(3); });
  std::cout << "mdspan=" << (OPENMX_CORE_STD_MDSPAN ? "std" : "Kokkos") << '\n';
}

void periodic_test() {
  for (int radius = 0; radius <= 12; ++radius) {
    int next = 1;
    std::vector<bool> seen(openmx_periodic_count(radius));
    for (int i = -radius; i <= radius; ++i)
      for (int j = -radius; j <= radius; ++j)
        for (int k = -radius; k <= radius; ++k) {
          const int expected = i == 0 && j == 0 && k == 0 ? 0 : next++;
          const int actual = openmx_periodic_index(radius,i,j,k);
          require(expected == actual && !seen.at(actual), "periodic enumeration mismatch");
          seen[actual] = true;
        }
    require(std::all_of(seen.begin(),seen.end(),[](bool value) {return value;}), "missing periodic cell");
    require(openmx_periodic_index(radius,radius+1,0,0) == -1, "out of bounds periodic index");
    require(openmx_periodic_index(radius,0,-radius-1,0) == -1, "out of bounds periodic index");
  }
  require(openmx_periodic_count(644) > 0, "largest valid radius rejected");
  require(openmx_periodic_count(645) == -1 && openmx_periodic_count(INT_MAX) == -1,
          "periodic count overflow not rejected");
  require(openmx_periodic_index(INT_MIN,0,0,0) == -1, "negative periodic radius accepted");
  require(openmx_periodic_index(644,644,644,644) == openmx_periodic_count(644)-1,
          "largest periodic index overflow");
}

void fft_grid_test() {
  // Independent oracle: enumerate all products of the supported prime powers.
  std::array<bool,4097> valid{};
  for (int a = 1; a <= 4096; a *= 2)
    for (int b = a; b <= 4096; b *= 3)
      for (int c = b; c <= 4096; c *= 5)
        for (int d = c; d <= 4096; d *= 7) valid[d] = true;
  for (int n = 0; n <= 4096; ++n)
    require(openmx_core_fft_grid_supported(n) == valid[n], "FFT grid factorization mismatch");
  require(!fft_grid_supported(INT_MIN) && !fft_grid_supported(INT_MAX), "FFT bounds failed");
  require(fft_grid_supported(1<<30), "large valid FFT grid rejected");
  require(std::string(openmx_core_version()) == "4.0 GPU", "application version changed");
}

void bessel_test() {
  for (int lmax : {0,1,4,12,29}) {
    for (double x : {0.0,1e-8,0.01,0.2,1.0,4.2,20.0,70.0}) {
      std::array<double,32> sb, derivative;
      sb.fill(-999); derivative.fill(-999);
      spherical_bessel(x,lmax,sb.data(),derivative.data());
      for (int l = 0; l <= lmax+1; ++l) near(sb[l],reference_bessel(l,x),2e-10);
      for (int l = 0; l <= lmax; ++l) {
        if (x == 0) near(derivative[l],l == 1 ? 1.0/3.0 : 0.0);
        else if (x >= 0.01) {
          const double step = 1e-5;
          const double numerical = (reference_bessel(l,x+step)-reference_bessel(l,x-step))/(2*step);
          near(derivative[l],numerical,3e-9);
        }
      }
    }
  }
  double values[32], derivatives[32];
  require(openmx_core_spherical_bessel(0,1,values,derivatives), "C Bessel rejected origin");
  near(values[2],0);
  near(derivatives[1],1.0/3.0);
  require(!openmx_core_spherical_bessel(-1,0,values,derivatives), "negative Bessel argument accepted");
  require(!openmx_core_spherical_bessel(1,30,values,derivatives), "excess Bessel order accepted");
  require(!openmx_core_spherical_bessel(std::numeric_limits<double>::infinity(),0,values,derivatives),
          "infinite Bessel argument accepted");
}

double radial_value(int l, int m, double r) {
  return (m+1)*std::pow(r,l)*std::exp(-r*(1+0.2*m));
}
RadialSamples samples(int k_count = 67, int intervals = 31) {
  const std::array<int,4> counts{2,0,1,1};
  return sample_radial({3,intervals,k_count,0.0,0.1,0.15},counts,radial_value);
}
void compare_reference(const RadialSamples& input, const Tensor<double,2>& output) {
  const auto grid = input.grid;
  for (std::size_t orbital = 0; orbital < input.orbitals.size(); ++orbital)
    for (int k = 0; k < grid.k_count; ++k) {
      double reference = 0;
      const auto info = input.orbitals[orbital];
      for (int i = 0; i <= grid.intervals; ++i) {
        const double r = grid.rmin+i*grid.radial_step;
        const double weighted = reference_bessel(info.angular, k*grid.k_step*r)*r*r;
        reference += radial_value(info.angular,info.multiplicity,r)*weighted*
                     (i == 0 || i == grid.intervals ? 0.5 : 1.0);
      }
      near(element(output.view(),orbital,k),reference*grid.radial_step,5e-11);
    }
}
void radial_host_test() {
  const auto input = samples();
  compare_reference(input,transform_host(input));
  compare_reference(samples(1,1),transform_host(samples(1,1)));
  require(transform_host(samples(0)).size() == 0, "empty k grid failed");
  const std::array<int,1> counts{0};
  auto empty = sample_radial({0,3,1,0,0.1,0.1},counts,radial_value);
  require(transform_host(empty).size() == 0, "empty orbital list failed");
  rejects([] { (void)samples(1,0); });
  rejects([] { (void)samples(-1); });
  rejects([] {
    const std::array<int,1> negative{-1};
    (void)sample_radial({0,1,1,0,0.1,0.1},negative,radial_value);
  });
  rejects([] {
    const std::array<int,1> count{1};
    (void)sample_radial({0,1,1,0,0.1,0.1},count,[](int,int,double) {
      return std::numeric_limits<double>::quiet_NaN();
    });
  });
}
int device() { return has_accelerator() ? 0 : -1; }
void cpu_request_test() {
  Context context;
  const auto input = samples();
  compare_reference(input,context.transform(input,-1));
  if (has_accelerator())
    require(!Kokkos::is_initialized(), "CPU request initialized the GPU runtime");
}
void radial_execution_test() {
  Context context;
  for (auto shape : {std::pair{67,31}, std::pair{5,9}, std::pair{33,1}, std::pair{0,31}}) {
    const auto input = samples(shape.first,shape.second);
    compare_reference(input,context.transform(input,device()));
    const auto expected = transform_host(input);
    const auto result = context.transform(input,device());
    for (std::size_t i = 0; i < result.size(); ++i) near(result.data()[i],expected.data()[i],2e-12);
  }
  // Independent contexts must not finalize the runtime owned by the first.
  { Context second; compare_reference(samples(),second.transform(samples(),device())); }
  require(Kokkos::is_initialized(), "secondary context finalized borrowed runtime");
  compare_reference(samples(),context.transform(samples(),device()));
}

struct CallbackState { int stored = 0; bool fail = false; };
double c_sample(void* user, int, int, double r) {
  return static_cast<CallbackState*>(user)->fail ? std::numeric_limits<double>::quiet_NaN() : std::exp(-r);
}
void c_store(void* user, int l, int m, int k, double value) {
  require(l == 0 && m == 0 && k >= 0 && std::isfinite(value), "invalid committed transform");
  ++static_cast<CallbackState*>(user)->stored;
}
int c_entry(int, char**, OpenMXCoreContext* context) {
  CallbackState state;
  int counts[] = {1};
  char error[128];
  require(openmx_core_pao_transform(context,device(),0,counts,10,35,0,0.1,0.1,
                                   &state,c_sample,c_store,error,sizeof(error)), error);
  require(state.stored == 35 && !error[0], "incorrect C output count");
  state = {0,true};
  require(!openmx_core_pao_transform(context,device(),0,counts,10,35,0,0.1,0.1,
                                    &state,c_sample,c_store,error,sizeof(error)), "bad sample accepted");
  require(state.stored == 0 && error[0], "failed transform partially committed");
  require(!openmx_core_pao_transform(context,-1,INT_MAX,nullptr,10,1,0,0.1,0.1,
                                    &state,c_sample,c_store,nullptr,0), "bad C dimensions accepted");
  openmx_core_finish(context);
  openmx_core_finish(context);
  state = {};
  require(!openmx_core_pao_transform(context,device(),0,counts,10,1,0,0.1,0.1,
                                    &state,c_sample,c_store,error,sizeof(error)), "finished context reused");
  require(state.stored == 0, "finished context committed values");
  return 0;
}
void lifecycle_test() {
  require(!Kokkos::is_initialized(), "runtime initialized before ownership scope");
  {
    Context context;
    compare_reference(samples(),context.transform(samples(),device()));
    require(Kokkos::is_initialized(), "execution did not initialize runtime");
  }
  require(!Kokkos::is_initialized() && Kokkos::is_finalized(), "runtime not released by RAII");
}
void borrowed_runtime_test() {
  Kokkos::ScopeGuard owner(Kokkos::InitializationSettings().set_disable_warnings(true));
  {
    Context context;
    compare_reference(samples(),context.transform(samples(),device()));
    context.finish();
  }
  require(Kokkos::is_initialized(), "external runtime was finalized");
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("specify a unit test case");
    const std::string name = argv[1];
    if (name == "tensor") tensor_test();
    else if (name == "periodic") periodic_test();
    else if (name == "fft-grid") fft_grid_test();
    else if (name == "bessel") bessel_test();
    else if (name == "radial-host") radial_host_test();
    else if (name == "cpu-request") cpu_request_test();
    else if (name == "radial-execution") radial_execution_test();
    else if (name == "c-api") require(openmx_core_run(argc,argv,c_entry) == 0, "C context scope failed");
    else if (name == "lifecycle") lifecycle_test();
    else if (name == "borrowed-runtime") borrowed_runtime_test();
    else throw std::invalid_argument("unknown unit test case");
    std::cout << name << ": passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
  }
}
