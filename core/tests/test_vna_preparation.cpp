#include <openmx/core/vna_preparation.hpp>
#include "openmx_core_vna_preparation.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
template<class F> void rejects(F&& function) {
  bool failed = false;
  try { function(); } catch (const std::exception&) { failed = true; }
  require(failed,"invalid VNA preparation accepted");
}

void layout_test()
{
  static_assert(!std::is_copy_constructible_v<VnaPreparation>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(
      element(std::declval<const VnaPreparation&>().weights(),0,0,0))>>);
  const int first[]{2,0,1}, last[]{0,2};
  const std::array species{SpeciesBasisView{2,first},SpeciesBasisView{-1,{}},SpeciesBasisView{1,last}};
  VnaPreparation work({2,2,2,2,7},species);
  const auto& layout = work.basis();
  const std::array<std::size_t,10> offsets{0,2,2,3,3,3,3,3,5,5};
  require(std::equal(offsets.begin(),offsets.end(),layout.offsets().begin()),"radial offsets changed species/group order");
  require(layout.radial_count() == 5 && layout.orbitals()[0] == 7 && layout.orbitals()[1] == 0 &&
          layout.orbitals()[2] == 6,"empty groups or magnetic counts are wrong");
  require(work.weights().extent(0) == 2 && work.weights().extent(1) == 5 &&
          work.weights().extent(2) == 7,"radial values have rectangular padding");
  require(work.projector_count() == 6 && work.projector_components() == 18,"projector count mismatch");
  for (int l = 0; l <= 2; ++l)
    for (int radial = 0; radial < 2; ++radial) {
      require(element(work.projectors(),0,2*l+radial) == l &&
              element(work.projectors(),1,2*l+radial) == radial,"projector order changed");
    }
  require(element(work.projectors(),0,6) == 0 && element(work.projectors(),1,7) == 0,
          "legacy projector padding is uninitialized");
  element(work.weights(),1,4,6) = 53;
  require(work.weights().data_handle()[69] == 53,"mdspan radial stride mismatch");
  for (int lmax = 0; lmax <= 14; ++lmax) {
    std::vector<int> counts(lmax+1,1);
    const std::array basis{SpeciesBasisView{lmax,counts}};
    VnaPreparation high({lmax,1,lmax,1,1},basis);
    require(high.basis().orbitals()[0] == (lmax+1)*(lmax+1) &&
            high.projector_components() == (lmax+1)*(lmax+1),"highest angular number is truncated");
  }
}

void ownership_test()
{
  for (int call = 0; call < 24; ++call) {
    std::vector<int> counts{2,0,1};
    std::array species{SpeciesBasisView{2,counts}};
    VnaPreparation first({2,2,1,3,1+call},species);
    VnaPreparation second({2,2,0,0,1},species);
    counts.assign(200,0); // May reallocate: no borrowed source metadata may remain.
    species[0].maximum_angular = -1;
    require(first.basis().maximum_angular()[0] == 2 && element(first.basis().counts(),0,0) == 2,
            "preparation retained borrowed metadata");
    element(first.weights(),0,0,0) = 59;
    require(element(second.weights(),0,0,0) == 0,"independent preparations share weights");
    auto copied = first.basis();
    require(copied.offsets().data() != first.basis().offsets().data() &&
            copied.radial_count() == 3,"basis layout copy retains internal pointers");
  }
  const std::array empty{SpeciesBasisView{-1,{}}};
  VnaPreparation none({0,1,0,0,5},empty);
  require(none.weights().size() == 0 && none.projector_count() == 0 &&
          none.projectors().extent(1) == 2,"empty preparation allocates numeric padding");
}

void failure_test()
{
  const int counts[]{1,1};
  const std::array valid{SpeciesBasisView{1,counts}};
  for (VnaPreparationShape shape : {VnaPreparationShape{-1,1,1,1,3},{15,1,1,1,3},
      {1,0,1,1,3},{1,1,-1,1,3},{1,1,15,1,3},{1,1,1,-1,3},{1,1,1,INT_MAX,3},
      {1,1,1,1,0},{1,1,1,1,-1}})
    rejects([&] { VnaPreparation work(shape,valid); });
  rejects([&] { VnaPreparation work({1,1,1,1,3},{}); });
  const int bad[]{-1,0}, excess[]{2,0}, huge[]{0,INT_MAX};
  for (auto input : {std::span<const int>(bad),std::span<const int>(excess),std::span<const int>(counts,1)}) {
    const std::array basis{SpeciesBasisView{1,input}};
    rejects([&] { VnaPreparation work({1,1,1,1,3},basis); });
  }
  const std::array overflow{SpeciesBasisView{1,huge}};
  rejects([&] { VnaPreparation work({1,INT_MAX,0,1,1},overflow); });
  const int too_many[]{INT_MAX/2+1};
  const std::array large_block{SpeciesBasisView{0,too_many}};
  rejects([&] { VnaPreparation work({0,INT_MAX,0,2,1},large_block); });
  const std::array wrong{SpeciesBasisView{2,counts}};
  rejects([&] { VnaPreparation work({1,1,1,1,3},wrong); });
}

void c_api_test()
{
  int angular[]{2,-1,1}, first[]{2,0,1}, last[]{0,2};
  int* rows[]{first,nullptr,last};
  const OpenMXVnaPreparationShape shape{2,2,2,2,7,3,angular,rows};
  OpenMXVnaPreparationBuffers views{}; char error[256];
  const auto destroy = &openmx_core_vna_preparation_destroy;
  std::unique_ptr<OpenMXVnaPreparation,decltype(destroy)> work(
      openmx_core_vna_preparation_create(&shape,&views,error,sizeof(error)),destroy);
  require(work != nullptr,error);
  require(views.radial_count == 5 && views.radial_elements == 35 && views.weighted[1] == views.weighted[0]+35,
          "C weight channels overlap or have padding");
  require(openmx_core_vna_radial_offset(&views,2,1,1) == 28,"C radial offset differs from mdspan layout");
  angular[0] = 0; first[0] = 0;
  require(views.maximum_angular[0] == 2 && views.basis_counts[0] == 2,"C views borrow source metadata");
  for (int channel = 0; channel < 2; ++channel)
    for (std::size_t i = 0; i < views.radial_elements; ++i) {
      require(views.weighted[channel][i] == 0,"uninitialized radial values");
      views.weighted[channel][i] = channel*100+i;
    }
  const auto retained = views;
  require(!openmx_core_vna_preparation_create(nullptr,&views,error,sizeof(error)) && error[0],"null shape accepted");
  require(!openmx_core_vna_preparation_create(&shape,nullptr,error,sizeof(error)),"null views accepted");
  auto invalid = shape; invalid.species_angular = nullptr;
  require(!openmx_core_vna_preparation_create(&invalid,&views,error,sizeof(error)) &&
          views.weighted[0] == retained.weighted[0] && views.weighted[1][34] == 134,
          "failed create changed retained output");
  const int none[]{-1};
  const OpenMXVnaPreparationShape empty{0,1,0,0,1,1,none,nullptr};
  OpenMXVnaPreparationBuffers vacant;
  auto* owner = openmx_core_vna_preparation_create(&empty,&vacant,error,sizeof(error));
  require(owner && !vacant.weighted[0] && !vacant.weighted[1] && vacant.projectors == 0 &&
          vacant.projector_angular[0] == 0,"empty C preparation is invalid");
  openmx_core_vna_preparation_destroy(owner);
  openmx_core_vna_preparation_destroy(nullptr);
}

void allocation_test()
{
  int angular[]{2,0}, a[]{2,0,1}, b[]{1};int* rows[]{a,b};
  const OpenMXVnaPreparationShape shape{2,2,2,2,7,2,angular,rows};
  OpenMXVnaPreparationBuffers views{};char error[256];
  openmx::test::fail_allocation_after(1000);
  auto* retained = openmx_core_vna_preparation_create(&shape,&views,error,sizeof(error));
  const long count = 1000-openmx::test::remaining_allocations();
  openmx::test::fail_allocation_after(-1);
  require(retained && count >= 8,"preparation allocations were not intercepted");
  const auto original = views;views.weighted[0][0] = 71;
  for (long failure = 0; failure < count; ++failure) {
    const auto before = openmx::test::live_allocations();
    openmx::test::fail_allocation_after(failure);
    auto* failed = openmx_core_vna_preparation_create(&shape,&views,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if (failed) openmx_core_vna_preparation_destroy(failed);
    require(!failed && error[0] && openmx::test::live_allocations() == before,"partial preparation leaked or escaped C");
    require(views.weighted[0] == original.weighted[0] && views.radial_offsets == original.radial_offsets &&
            views.weighted[0][0] == 71,"failed allocation published partial views");
  }
  openmx_core_vna_preparation_destroy(retained);
}
} // namespace

int main(int argc,char** argv)
{
  try {
    const std::string name = argc == 2 ? argv[1] : "";
    if (name == "layout") layout_test();
    else if (name == "ownership") ownership_test();
    else if (name == "failure") failure_test();
    else if (name == "c-api") c_api_test();
    else if (name == "allocation") allocation_test();
    else throw std::runtime_error("unknown VNA preparation test");
    std::cout << "PASS VNA preparation " << name << '\n';return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n';return 1; }
}
