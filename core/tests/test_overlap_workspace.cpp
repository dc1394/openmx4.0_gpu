#include <openmx/core/overlap_workspace.hpp>
#include "openmx_core_overlap.h"
#include "allocation_faults.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;

void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
template<class F> void rejects(F&& function) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  require(rejected,"invalid overlap shape accepted");
}

void layout_test()
{
  static_assert(!std::is_copy_constructible_v<OverlapWorkspace>);
  // All l=0..14 entries fit consecutively, including the highest magnetic
  // state and every radial multiplicity. No unused magnetic columns remain.
  for (int multiplicity : {1,2,5}) {
    std::size_t expected = 0;
    for (int l = 0; l <= 14; ++l)
      for (int radial = 0; radial < multiplicity; ++radial)
        for (int magnetic = 0; magnetic <= 2*l; ++magnetic)
          require(openmx_core_overlap_orbital(multiplicity,l,radial,magnetic) == expected++,
                  "orbital packing has an alias or gap");
    require(expected == 225*static_cast<std::size_t>(multiplicity),"orbital count mismatch");
  }
  OverlapWorkspace work({3,2,7,0.1,1.6,2});
  require(work.angular().extent(1) == 32 && work.angular().extent(2) == 32,
          "angular tensor has rectangular padding");
  require(work.conversion().extent(1) == 7 && work.conversion().extent(2) == 7,
          "conversion magnetic capacity mismatch");
  require(work.radial().extent(1) == 8 && work.radial().extent(2) == 7,
          "RF table shape mismatch");
  require(work.sums().extent(1) == 7 && work.sums().extent(2) == 64,
          "radial integral shape mismatch");
  for (std::size_t point = 0; point < work.wavevectors().size(); ++point)
    require(work.wavevectors()[point] == 0.1+static_cast<double>(point)*0.25,
            "wavevector construction changed order");
  for (std::size_t i = 0; i < work.angular().size(); ++i)
    require(work.angular().data_handle()[i].r == 0 && work.angular().data_handle()[i].i == 0,
            "angular storage is uninitialized");
  element(work.angular(),7,31,31) = {71,-83};
  require(work.angular().data_handle()[work.angular().size()-1].r == 71,"mdspan row layout mismatch");
  element(work.radial(),1,7,6) = 43;
  require(work.radial().data_handle()[work.radial().size()-1] == 43,"RF tensor stride mismatch");
}

void ownership_test()
{
  for (int call = 0; call < 16; ++call) {
    OverlapWorkspace first({call%4,1+call%3,5+call%3,0.,2.,2});
    OverlapWorkspace second({0,1,2,0.2,0.8,0});
    require(first.angular().data_handle() != second.angular().data_handle(),"workspaces share numeric storage");
    element(first.angular(),0,0,0) = {123,-7};
    first.sums().data_handle()[0] = 31;
    first.offsets().data_handle()[0] = 4;
    const auto* grid = first.wavevectors().data();
    const auto retained = first.sample(0.5,0);
    const auto again = first.sample(0.5,0);
    require(retained.values.data_handle() == again.values.data_handle(),"overlap Bessel cache copied data");
    first.sample(0.75,0);
    first.sample(1.25,0); // Cache full: reusable scratch must not invalidate cached data.
    require(first.wavevectors().data() == grid && first.angular().data_handle()[0].r == 123 &&
            first.sums().data_handle()[0] == 31 && first.offsets().data_handle()[0] == 4,
            "Bessel sampling moved or overwrote the borrowed work buffers");
    const double x = grid[1]*0.5;
    require(std::abs(element(retained.values,0,0,1)-std::sin(x)/x) < 1e-13,
            "cached Bessel view was invalidated");
    require(first.statistics().hits == 1 && first.statistics().misses == 3,"overlap cache statistics mismatch");
    require(second.angular().data_handle()[0].r == 0 && second.sums().data_handle()[0] == 0,
            "independent workspace was modified");
  }
}

void failure_test()
{
  const OverlapShape valid{2,2,9,0.,2.,2};
  for (int bad : {-1,15,INT_MAX}) {
    auto shape = valid; shape.maximum_angular = bad;
    rejects([&] { OverlapWorkspace work(shape); });
  }
  for (int bad : {-1,0,INT_MAX}) {
    auto shape = valid; shape.multiplicity = bad;
    rejects([&] { OverlapWorkspace work(shape); });
  }
  for (int bad : {-1,0,1,INT_MAX}) {
    auto shape = valid; shape.points = bad;
    rejects([&] { OverlapWorkspace work(shape); });
  }
  for (double bad : {-1.,std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()}) {
    auto shape = valid; shape.kmin = bad;
    rejects([&] { OverlapWorkspace work(shape); });
    shape = valid; shape.kmax = bad;
    rejects([&] { OverlapWorkspace work(shape); });
  }
  rejects([] { OverlapWorkspace work({0,1,2,1.,1.,0}); });
  rejects([] { OverlapWorkspace work({0,1,2,2.,1.,0}); });
  // A valid radial offset alone is insufficient: l*combo and l*points
  // also have to fit the legacy int arithmetic before any allocation.
  rejects([] { OverlapWorkspace work({14,1000,2,0.,1.,0}); });
  rejects([] { OverlapWorkspace work({14,1,INT_MAX/20,0.,1.,0}); });
}

void c_api_test()
{
  const OpenMXOverlapShape shape{2,2,9,0.1,2.1,2};
  OpenMXOverlapBuffers views{};
  char error[256];
  const auto destroy = &openmx_core_overlap_destroy;
  std::unique_ptr<OpenMXOverlapWorkspace,decltype(destroy)> work(
      openmx_core_overlap_create(&shape,&views,error,sizeof(error)),destroy);
  require(work != nullptr,error);
  require(views.orbital_capacity == 18 && views.magnetic_capacity == 5 &&
          views.radial_elements == 54 && views.sum_elements == 180 && views.max_combinations == 36,
          "C view capacities mismatch");
  for (int channel = 0; channel < 8; ++channel) {
    for (int l0 = 0; l0 <= shape.maximum_angular; ++l0)
      for (int r0 = 0; r0 < shape.multiplicity; ++r0)
        for (int m0 = 0; m0 <= 2*l0; ++m0)
          for (int l1 = 0; l1 <= shape.maximum_angular; ++l1)
            for (int r1 = 0; r1 < shape.multiplicity; ++r1)
              for (int m1 = 0; m1 <= 2*l1; ++m1) {
                const auto index = openmx_core_overlap_pair(&views,l0,r0,m0,l1,r1,m1);
                require(index < 18*18 && views.angular[channel][index].r == 0,
                        "C pair mapping aliases another pair");
                views.angular[channel][index] = {static_cast<double>(1+index),static_cast<double>(channel)};
              }
    for (std::size_t i = 0; i < 18*18; ++i)
      require(views.angular[channel][i].r == static_cast<double>(1+i) &&
              views.angular[channel][i].i == channel,"C packed channel has a gap or wrong stride");
    if (channel)
      require(views.angular[channel] == views.angular[0]+channel*18*18 &&
              views.conversion[channel] == views.conversion[0]+channel*5*5,"complex channels overlap");
  }
  for (int c = 0; c < 4; ++c) {
    require(views.sums[c] == views.sums[0]+c*180,"sum channels overlap");
    require(std::all_of(views.sums[c],views.sums[c]+180,[](double x) {return x == 0;}),"stale sum storage");
  }
  require(views.radial[1] == views.radial[0]+54 && views.offsets[1] == views.offsets[0]+36,
          "RF/offset channels overlap");
  OpenMXPairBesselTable table{};
  require(openmx_core_overlap_sample(work.get(),0.5,4,&table,error,sizeof(error)),error);
  const auto saved = table;
  require(!openmx_core_overlap_sample(work.get(),-1.,4,&table,error,sizeof(error)) &&
          table.values == saved.values && table.derivatives == saved.derivatives && error[0],
          "failed C sample published output");
  require(!openmx_core_overlap_sample(nullptr,1.,0,&table,error,sizeof(error)),"null workspace accepted");
  require(!openmx_core_overlap_sample(work.get(),1.,0,nullptr,error,sizeof(error)),"null sample output accepted");
  const auto old_views = views;
  require(!openmx_core_overlap_create(nullptr,&views,error,sizeof(error)) &&
          views.angular[0] == old_views.angular[0],"failed C create changed existing views");
  require(!openmx_core_overlap_create(&shape,nullptr,error,sizeof(error)),"null C views accepted");
  auto bad = shape; bad.multiplicity = INT_MAX;
  require(!openmx_core_overlap_create(&bad,&views,error,sizeof(error)) && error[0] &&
          views.angular[0] == old_views.angular[0],"oversized create published partial buffers");
  long hits = -1, misses = -1;
  openmx_core_overlap_statistics(work.get(),&hits,&misses);
  require(hits == 0 && misses == 1,"C overlap statistics mismatch");
  openmx_core_overlap_statistics(nullptr,&hits,&misses);
  require(hits == 0 && misses == 0,"null C statistics mismatch");
  openmx_core_overlap_statistics(work.get(),nullptr,nullptr);
  openmx_core_overlap_destroy(nullptr);
}

void allocation_test()
{
  const OpenMXOverlapShape shape{2,2,9,0.1,2.1,2};
  OpenMXOverlapBuffers views{};
  char error[256];
  openmx::test::fail_allocation_after(1000);
  auto* retained = openmx_core_overlap_create(&shape,&views,error,sizeof(error));
  const long allocations = 1000-openmx::test::remaining_allocations();
  openmx::test::fail_allocation_after(-1);
  require(retained != nullptr && allocations >= 8,"allocation fault test did not intercept core allocations");
  const auto original = views;
  views.angular[0][0] = {41,-7};
  for (long fail = 0; fail < allocations; ++fail) {
    const long before = openmx::test::live_allocations();
    openmx::test::fail_allocation_after(fail);
    auto* failed = openmx_core_overlap_create(&shape,&views,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if (failed) openmx_core_overlap_destroy(failed);
    require(failed == nullptr && error[0],"allocation failure escaped the C boundary");
    require(openmx::test::live_allocations() == before,"partial overlap allocation leaked");
    require(views.angular[0] == original.angular[0] && views.radial[0] == original.radial[0] &&
            views.wavevectors == original.wavevectors && views.orbital_capacity == original.orbital_capacity &&
            views.angular[0][0].r == 41,"failed allocation invalidated existing views");
  }
  openmx_core_overlap_destroy(retained);
}

} // namespace

int main(int argc, char** argv)
{
  try {
    const std::string name = argc == 2 ? argv[1] : "";
    if (name == "layout") layout_test();
    else if (name == "ownership") ownership_test();
    else if (name == "failure") failure_test();
    else if (name == "c-api") c_api_test();
    else if (name == "allocation") allocation_test();
    else throw std::runtime_error("unknown overlap workspace test");
    std::cout << "PASS overlap workspace " << name << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
