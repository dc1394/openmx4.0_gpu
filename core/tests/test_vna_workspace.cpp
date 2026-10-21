#include <openmx/core/vna_workspace.hpp>
#include "openmx_core_vna.h"
#include "allocation_faults.hpp"

#include <algorithm>
#include <climits>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }

void layout_test()
{
  static_assert(!std::is_copy_constructible_v<VnaWorkspace>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(
      element(std::declval<const VnaWorkspace&>().angular(),0,0,0))>>);
  for (int lmax = 0; lmax <= 14; ++lmax) {
    VnaWorkspace work({lmax,1,3});
    const auto orbitals = static_cast<std::size_t>((lmax+1)*(lmax+1));
    require(work.angular().extent(0) == 4 && work.angular().extent(1) == orbitals &&
            work.angular().extent(2) == orbitals,"angular workspace has magnetic padding");
    require(work.conversion().extent(1) == 2*lmax+1 &&
            work.bessel().extent(1) == 2*lmax+1 && work.bessel().extent(2) == 3 &&
            work.scratch().extent(1) == 2*lmax+3,"VNA scratch extents are wrong");
    element(work.angular(),3,orbitals-1,orbitals-1) = {19,-23};
    require(work.angular().data_handle()[work.angular().size()-1].i == -23,"mdspan angular layout mismatch");
    element(work.bessel(),1,2*lmax,2) = 29;
    require(work.bessel().data_handle()[work.bessel().size()-1] == 29,"Bessel stride mismatch");
  }
}

void ownership_test()
{
  for (int invocation = 0; invocation < 24; ++invocation) {
    VnaWorkspace first({invocation%4,1+invocation%3,1+invocation});
    VnaWorkspace second({0,1,1});
    require(first.angular().data_handle() != second.angular().data_handle(),"workspaces share state");
    first.angular().data_handle()[0] = {41,-7};
    first.conversion().data_handle()[0] = {31,11};
    first.bessel().data_handle()[0] = 17;
    first.scratch().data_handle()[0] = 5;
    require(second.angular().data_handle()[0].r == 0 && second.conversion().data_handle()[0].r == 0 &&
            second.bessel().data_handle()[0] == 0 && second.scratch().data_handle()[0] == 0,
            "independent workspace was overwritten");
  }
}

void failure_test()
{
  for (VnaShape shape : {VnaShape{-1,1,3},{15,1,3},{INT_MAX,1,3},
                        {1,0,3},{1,-1,3},{1,INT_MAX,3},
                        {1,1,0},{1,1,-1},{14,1,INT_MAX/20}}) {
    bool failed = false;
    try { VnaWorkspace work(shape); } catch (const std::exception&) { failed = true; }
    require(failed,"invalid or overflowing VNA workspace accepted");
  }
}

void c_api_test()
{
  const OpenMXVnaShape shape{2,2,7};
  OpenMXVnaBuffers views{};
  char error[256];
  const auto destroy = &openmx_core_vna_destroy;
  std::unique_ptr<OpenMXVnaWorkspace,decltype(destroy)> owner(
      openmx_core_vna_create(&shape,&views,error,sizeof(error)),destroy);
  require(owner != nullptr,error);
  require(views.orbital_capacity == 18 && views.magnetic_capacity == 5 &&
          views.bessel_orders == 5 && views.points == 7 && views.scratch_capacity == 7,
          "VNA C capacities differ from the owning tensors");
  for (int channel = 0; channel < 4; ++channel) {
    for (int l0 = 0; l0 <= 2; ++l0)
      for (int r0 = 0; r0 < 2; ++r0)
        for (int m0 = 0; m0 <= 2*l0; ++m0)
          for (int l1 = 0; l1 <= 2; ++l1)
            for (int r1 = 0; r1 < 2; ++r1)
              for (int m1 = 0; m1 <= 2*l1; ++m1) {
                const auto index = openmx_core_vna_pair(&views,l0,r0,m0,l1,r1,m1);
                require(index < 18*18 && views.angular[channel][index].r == 0,"VNA pair aliases another pair");
                views.angular[channel][index] = {1.+index,static_cast<double>(channel)};
              }
    for (std::size_t i = 0; i < 18*18; ++i)
      require(views.angular[channel][i].r == 1.+i && views.angular[channel][i].i == channel,
              "C orbital mapping has a gap or wrong channel");
    require(views.angular[channel] == views.angular[0]+channel*18*18 &&
            views.conversion[channel] == views.conversion[0]+channel*25,"complex channels overlap");
  }
  require(views.bessel[1] == views.bessel[0]+35 && views.scratch[1] == views.scratch[0]+7,
          "Bessel value and derivative buffers overlap");
  views.bessel[0][34] = 43; views.bessel[1][0] = 47;
  require(views.bessel[0][34] == 43,"Bessel channel was overwritten");
  const auto retained = views;
  require(!openmx_core_vna_create(nullptr,&views,error,sizeof(error)) && error[0],"null shape accepted");
  require(!openmx_core_vna_create(&shape,nullptr,error,sizeof(error)),"null views accepted");
  auto invalid = shape; invalid.points = -1;
  require(!openmx_core_vna_create(&invalid,&views,error,sizeof(error)) && error[0] &&
          views.angular[0] == retained.angular[0] && views.bessel[0][34] == 43,
          "failed create changed retained buffers");
  openmx_core_vna_destroy(nullptr);
}

void allocation_test()
{
  const OpenMXVnaShape shape{2,2,7};
  OpenMXVnaBuffers views{};
  char error[256];
  openmx::test::fail_allocation_after(1000);
  auto* owner = openmx_core_vna_create(&shape,&views,error,sizeof(error));
  const long count = 1000-openmx::test::remaining_allocations();
  openmx::test::fail_allocation_after(-1);
  require(owner && count >= 5,"VNA allocations were not intercepted");
  const auto retained = views;
  views.angular[0][0] = {59,61}; views.bessel[1][0] = 67;
  for (long fail = 0; fail < count; ++fail) {
    const auto before = openmx::test::live_allocations();
    openmx::test::fail_allocation_after(fail);
    auto* failed = openmx_core_vna_create(&shape,&views,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if (failed) openmx_core_vna_destroy(failed);
    require(!failed && error[0],"allocation failure escaped the C boundary");
    require(openmx::test::live_allocations() == before,"partial VNA workspace leaked");
    require(views.angular[0] == retained.angular[0] && views.bessel[1] == retained.bessel[1] &&
            views.orbital_capacity == retained.orbital_capacity &&
            views.angular[0][0].r == 59 && views.bessel[1][0] == 67,"failed create published partial views");
  }
  openmx_core_vna_destroy(owner);
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
    else throw std::runtime_error("unknown VNA workspace test");
    std::cout << "PASS VNA workspace " << name << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
