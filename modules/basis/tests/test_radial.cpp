#include <openmx/basis/radial_table.hpp>
#include <openmx/basis/radial.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

// A test-local allocation gate. The production static library has no mutable
// global state. Disable only while exercising already constructed objects.
namespace { std::atomic<long> allocations_before_failure{-1}; }
void* operator new(std::size_t bytes)
{
  auto remaining = allocations_before_failure.load();
  if (remaining >= 0) {
    if (remaining == 0) throw std::bad_alloc();
    allocations_before_failure.store(remaining-1);
  }
  if (auto* p = std::malloc(bytes ? bytes : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace openmx::basis;
void require(bool condition, const char* text)
{
  if (!condition) throw std::runtime_error(text);
}
void near(double value, double expected, const std::string& context = "")
{
  if (std::isfinite(value) && std::abs(value-expected) <= 3e-13*(1+std::abs(expected))) return;
  std::ostringstream text;
  text << std::setprecision(17) << context << ": expected " << expected << ", got " << value;
  throw std::runtime_error(text.str());
}
template<class Function> void rejects(Function&& function)
{
  bool caught = false;
  try { function(); } catch (const std::exception&) { caught = true; }
  require(caught,"invalid input was accepted");
}
const std::array<double,8> radii{.15,.3,.6,1.1,1.7,2.6,3.8,5.2};
const std::array<double,8> values{.7,1.4,-.2,.9,-.4,.3,.1,0};

void radial()
{
  const std::array<double,15> points{0,.1,.15,.2,.3,.4,.6,.85,1.1,1.5,2,3,4.5,5.2,5.3};
  // RadialF/Nonlocal_RadialF/VNAF at 170c51d3, contraction disabled.
  // Frozen prior to this extraction. The PAO first-cell extrapolation is
  // deliberate compatibility behavior, including at the first mesh point.
  const std::array<double,15> pao{.7,.7,-.36781250000000026,.62259259259259359,
    1.4,1.1001851851851847,-.2,.17978219696969672,.9,.034500561167227745,
    -.37523809523809531,.32536969203635868,.032051282051282041,0,0};
  auto projector = pao;
  projector[0] = -.99999999999999911; projector[1] = .24444444444444519;
  projector[2] = .70000000000000018; projector[3] = 1.0444444444444445;
  auto neutral = pao;
  neutral[0] = .44055555555555626; neutral[1] = .43764705882353011;
  const std::array<std::array<double,15>,4> reference{pao,projector,neutral,neutral};
  const std::array<RadialKind,4> kinds{RadialKind::pao,RadialKind::projector,
    RadialKind::neutral_potential,RadialKind::neutral_projector};
  const RadialChannel channel{0,values};
  for (std::size_t k = 0; k < kinds.size(); ++k) {
    const RadialTable table(kinds[k],radii,{&channel,1},radii.back());
    const RadialTableView view(kinds[k],radii,{&channel,1},radii.back());
    for (std::size_t i = 0; i < points.size(); ++i) {
      const auto location = "kind="+std::to_string(k)+" channel=0 r="+std::to_string(points[i]);
      near(table.sample(0,points[i]),reference[k][i],"owner "+location);
      near(view.sample(0,points[i]),reference[k][i],"view "+location);
      near(sample_radial_row(kinds[k],0,radii,values,radii.back(),points[i]),reference[k][i],
           "scalar "+location);
    }
  }
  for (int l : {1,2,4,29}) {
    for (auto kind : {RadialKind::pao,RadialKind::projector}) {
      near(sample_radial_row(kind,l,radii,values,0,0),0);
      near(sample_radial_row(kind,l,radii,values,0,5.3),0);
    }
  }
  const RadialTable cutoff(RadialKind::neutral_potential,radii,{&channel,1},2.6);
  near(cutoff.sample(0,2.6),values[5]);
  near(cutoff.sample(0,std::nextafter(2.6,3.0)),0);
  // Analytic, independent oracle for the interior and first-cell stencil.
  std::array<double,8> quadratic;
  for (std::size_t i = 0; i < radii.size(); ++i)
    quadratic[i] = 2+3*radii[i]+4*radii[i]*radii[i];
  const RadialChannel analytic{0,quadratic};
  const RadialTable polynomial(RadialKind::pao,radii,{&analytic,1});
  for (double r : {.15,.2,.5,.9,1.2,1.9,2.4}) near(polynomial.sample(0,r),2+3*r+4*r*r);
  // Minimum-size tables also exercise the last-point ghost stencil.
  const std::array<double,3> small_r{0,1,2}, constant{2,2,2};
  for (auto kind : {RadialKind::pao,RadialKind::projector})
    for (double r : {0.,.1,1.,1.9,2.}) near(sample_radial_row(kind,0,small_r,constant,0,r),2);
}

void reciprocal()
{
  const double step = 2.4/8;
  const std::array<double,11> points{-.1,0,.05,.15,.3,.45,.6,1,1.9,2.1,2.11};
  // RF_BesselF/NLRF_BesselF at 6c681182. k=0 uses the second stencil.
  const std::array<double,11> expected{-.044444444444445286,-4.3000000000000016,
    -2.2064814814814828,.39999999999999902,1.3999999999999995,.5749999999999994,
    -.20000000000000009,.57037037037036997,.055555555555555525,0,0};
  for (auto kind : {ReciprocalKind::pao,ReciprocalKind::projector}) {
    for (std::size_t i = 0; i < points.size(); ++i)
      near(sample_reciprocal_row(kind,values,step,points[i]),expected[i],
           "reciprocal kind="+std::to_string(static_cast<int>(kind))+" k="+std::to_string(points[i]));
    const std::array<double,1> single{3};
    near(sample_reciprocal_row(kind,single,step,0),3);
    near(sample_reciprocal_row(kind,single,step,.01),0);
    const std::array<double,2> pair{2,4};
    near(sample_reciprocal_row(kind,pair,step,-1),2);
    near(sample_reciprocal_row(kind,pair,step,step/2),3);
    near(sample_reciprocal_row(kind,pair,step,step),4);
    near(sample_reciprocal_row(kind,pair,step,step*2),0);
  }
  near(sample_reciprocal_row(ReciprocalKind::projector,values,1e-16,.5e-16),values[1]);
  near(sample_reciprocal_row(ReciprocalKind::pao,values,1e-16,.5e-16),.4);
}

void ownership()
{
  auto grid = radii, first = values, second = values;
  for (auto& v : second) v *= -2;
  const std::array<RadialChannel,2> channels{{{0,first},{2,second}}};
  RadialTable table(RadialKind::pao,grid,channels);
  const RadialTableView borrowed(RadialKind::pao,grid,channels);
  auto copy = table;
  auto moved = std::move(copy);
  copy = moved;
  moved = std::move(copy);
  first[2] = 7;
  near(borrowed.sample(0,.6),7);
  near(table.sample(0,.6),-.2);
  grid.fill(-1); first.fill(-1); second.fill(-1);
  for (const auto* owner : {&table,&moved}) {
    require(owner->channel_count() == 2 && owner->angular(1) == 2,"lost channel metadata");
    near(owner->sample(0,.85),.17978219696969672);
    near(owner->sample(1,.85),-2*.17978219696969672);
  }
  const RadialTable empty(RadialKind::pao,radii,{});
  require(empty.channel_count() == 0,"empty table grew channels");
  rejects([&] { (void)empty.sample(0,1); });
}

void invalid()
{
  const RadialChannel channel{0,values};
  const RadialTable table(RadialKind::pao,radii,{&channel,1});
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (double coordinate : {-1.,nan,std::numeric_limits<double>::infinity()})
    rejects([&] { (void)table.sample(0,coordinate); });
  rejects([&] { (void)table.sample(1,0); });
  rejects([&] { (void)table.angular(1); });
  rejects([&] { RadialTable bad(static_cast<RadialKind>(-1),radii,{&channel,1}); });
  auto grid = radii;
  for (double bad : {-1.,nan,.3}) {
    grid[2] = bad;
    rejects([&] { RadialTable t(RadialKind::pao,grid,{&channel,1}); });
  }
  auto row = values; row.back() = nan;
  const RadialChannel invalid_row{0,row}, invalid_l{30,values};
  rejects([&] { RadialTable t(RadialKind::pao,radii,{&invalid_row,1}); });
  rejects([&] { RadialTable t(RadialKind::pao,radii,{&invalid_l,1}); });
  rejects([&] { RadialTable t(RadialKind::neutral_potential,radii,{&channel,1},6); });
  const RadialChannel short_row{0,{values.data(),3}};
  rejects([&] { RadialTable t(RadialKind::pao,radii,{&short_row,1}); });
  for (auto kind : {ReciprocalKind::pao,ReciprocalKind::projector}) {
    rejects([&] { (void)sample_reciprocal_row(kind,{},.1,0); });
    for (double step : {0.,-.1,nan,std::numeric_limits<double>::max()})
      rejects([&] { (void)sample_reciprocal_row(kind,values,step,0); });
  }
}

void concurrent()
{
  const RadialChannel channel{0,values};
  const RadialTable shared(RadialKind::pao,radii,{&channel,1});
  std::array<std::future<void>,4> workers;
  for (std::size_t i = 0; i < workers.size(); ++i)
    workers[i] = std::async(std::launch::async,[&,i] {
      auto independent = values;
      for (auto& v : independent) v *= i+1;
      const RadialChannel row{0,independent};
      const RadialTable local(RadialKind::pao,radii,{&row,1});
      for (int repeat = 0; repeat < 1000; ++repeat) {
        near(shared.sample(0,.85),.17978219696969672);
        near(local.sample(0,.85),(i+1)*.17978219696969672);
      }
    });
  for (auto& worker : workers) worker.get();
}

void allocation()
{
  const RadialChannel channel{0,values};
  const RadialTable source(RadialKind::pao,radii,{&channel,1});
  RadialTable target(RadialKind::projector,radii,{&channel,1});
  // Copy assignment must keep the entire old table if any allocation fails.
  long failures = 0;
  for (long index = 0; index < 16; ++index) {
    bool failed = false;
    allocations_before_failure = index;
    try { target = source; } catch (const std::bad_alloc&) { failed = true; }
    allocations_before_failure = -1;
    if (!failed) break;
    ++failures;
    near(target.sample(0,0),-.99999999999999911);
  }
  require(failures >= 3,"copy allocation fault sweep missed owned buffers");
  near(target.sample(0,0),.7);
  const int angular = 0;
  char error[128];
  failures = 0;
  for (long index = 0; index < 16; ++index) {
    allocations_before_failure = index;
    auto* owner = openmx_basis_radial_table_create(OPENMX_BASIS_RADIAL_PAO,
        radii.size(),radii.data(),1,&angular,values.size(),values.data(),0,error,sizeof error);
    allocations_before_failure = -1;
    if (owner) {
      openmx_basis_radial_table_destroy(owner);
      break;
    }
    ++failures;
    require(error[0] != '\0',"allocation failure lost its diagnostic");
  }
  require(failures >= 5,"C owner allocation fault sweep missed construction");
  // Successful scalar and retained-table sampling must allocate nothing.
  allocations_before_failure = 0;
  const double scalar = sample_radial_row(RadialKind::pao,0,radii,values,0,.85);
  const double owned = target.sample(0,.85);
  double c_scalar = 0;
  const int ok = openmx_basis_radial_sample(OPENMX_BASIS_RADIAL_PAO,0,
      radii.size(),radii.data(),values.data(),0,.85,&c_scalar,error,sizeof error);
  allocations_before_failure = -1;
  near(scalar,.17978219696969672); near(owned,scalar); near(c_scalar,scalar);
  require(ok,"scalar sampling attempted an allocation");
}
} // namespace

int main(int argc, char** argv)
{
  try {
    if (argc != 2) throw std::invalid_argument("expected a test case");
    const std::string name = argv[1];
    if (name == "radial") radial();
    else if (name == "reciprocal") reciprocal();
    else if (name == "ownership") ownership();
    else if (name == "invalid") invalid();
    else if (name == "concurrent") concurrent();
    else if (name == "allocation") allocation();
    else throw std::invalid_argument("unknown test case");
    std::cout << "PASS basis " << name << '\n';
    return 0;
  } catch (const std::exception& error) {
    allocations_before_failure = -1;
    std::cerr << error.what() << '\n';
    return 1;
  }
}
