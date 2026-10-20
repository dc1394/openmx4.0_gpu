#include <openmx/core/hvna_workspace.hpp>
#include "openmx_core_hvna_workspace.h"
#include "allocation_faults.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool value,const char* reason) { if (!value) throw std::runtime_error(reason); }
using ProjectorOwner=std::unique_ptr<OpenMXHvnaProjectors,decltype(&openmx_core_hvna_projectors_destroy)>;
using WorkspaceOwner=std::unique_ptr<OpenMXHvnaWorkspace,decltype(&openmx_core_hvna_workspace_destroy)>;
struct Fixture {
  int angular[2]{2,1}, first[3]{2,0,1}, second[2]{0,2};
  int* counts[2]{first,second};
  double values[2][2][2]{{{1,-2},{3,4}},{{5,6},{-7,8}}};
  double* channels[2][2]{{values[0][0],values[0][1]},{values[1][0],values[1][1]}};
  double** energies[2]{channels[0],channels[1]};
  OpenMXVnaPreparationBuffers basis{};
  OpenMXHvnaProjectorBuffers table{};
  OpenMXHvnaWorkspaceBuffers work{};
  char error[256]{};
  std::unique_ptr<OpenMXVnaPreparation,decltype(&openmx_core_vna_preparation_destroy)> preparation{
      nullptr,openmx_core_vna_preparation_destroy};
  Fixture() {
    const OpenMXVnaPreparationShape shape{2,2,1,2,3,2,angular,counts};
    preparation.reset(openmx_core_vna_preparation_create(&shape,&basis,error,sizeof(error)));
    require(bool(preparation),error);
  }
  OpenMXHvnaProjectors* create() {
    return openmx_core_hvna_projectors_create(preparation.get(),energies,&table,error,sizeof(error));
  }
};

void layout_test()
{
  static_assert(!std::is_copy_constructible_v<HvnaWorkspace>);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(
      element(std::declval<const HvnaWorkspace&>().matrix(),0,0))>>);
  const int first[]{2,0,1},second[]{0,2};
  const std::array basis{SpeciesBasisView{2,first},SpeciesBasisView{1,second}};
  VnaPreparation prep({2,2,1,2,3},basis);
  const std::array<double,8> energy{1,-2,3,4,5,6,-7,8};
  HvnaProjectors projectors(prep,energy);HvnaWorkspace work(projectors);
  require(projectors.groups()==4 && projectors.components()==8,"projector dimensions");
  require(projectors.orbitals()[0]==7 && projectors.orbitals()[1]==6,"ragged species orbitals");
  require(work.matrix().extent(0)==7 && work.matrix().extent(1)==7,"workspace uses padded basis capacity");
  require(projectors.lengths()[0]==1 && projectors.lengths()[2]==3,"group magnetic lengths");
  require(element(projectors.energies(),1,2)==-7,"energy mdspan layout");
  element(work.matrix(),6,5)=43;
  require(work.matrix().data_handle()[6*7+5]==43,"matrix mdspan stride");
  bool rejected=false;
  try { HvnaProjectors bad(prep,{}); } catch(const std::invalid_argument&) { rejected=true; }
  require(rejected,"wrong energy table size accepted");
}

void ownership_test()
{
  WorkspaceOwner retained(nullptr,openmx_core_hvna_workspace_destroy);OpenMXHvnaWorkspaceBuffers old{};
  for (int call=0;call<12;++call) {
    Fixture f;ProjectorOwner projectors(f.create(),openmx_core_hvna_projectors_destroy);
    require(bool(projectors),f.error);
    WorkspaceOwner next(openmx_core_hvna_workspace_create(projectors.get(),&f.work,f.error,sizeof(f.error)),
        openmx_core_hvna_workspace_destroy);require(bool(next),f.error);
    f.values[1][1][0]=99;f.first[0]=0;f.preparation.reset();
    require(f.table.energies[6]==-7 && f.table.orbitals[0]==7,"projectors borrowed input metadata");
    projectors.reset();f.work.matrix[48]=call;
    if (retained)require(old.matrix[48]==call-1 && old.matrix!=f.work.matrix,"workspaces share memory");
    retained=std::move(next);old=f.work;
  }
  require(old.matrix[48]==11,"workspace retained a borrowed projector table");
}

void empty_test()
{
  const int count[]{1};
  for (int active=0;active<2;++active)for (int groups=0;groups<2;++groups) {
    const std::array basis{SpeciesBasisView{active?0:-1,active?std::span<const int>(count):std::span<const int>{}}};
    VnaPreparation prep({0,1,0,groups,1},basis);
    const std::array<double,1> energy{3};
    HvnaProjectors table(prep,{energy.data(),static_cast<std::size_t>(groups)});
    HvnaWorkspace work(table);
    require(work.matrix().size()==static_cast<std::size_t>(active),"empty basis allocates numeric padding");
    require(table.groups()==groups && table.energies().size()==static_cast<std::size_t>(groups),"empty projectors shape");
  }
  int angular[]{-1};const OpenMXVnaPreparationShape shape{0,1,0,0,1,1,angular,nullptr};
  OpenMXVnaPreparationBuffers basis{};OpenMXHvnaProjectorBuffers table{};OpenMXHvnaWorkspaceBuffers work{};
  char error[256];auto* prep=openmx_core_vna_preparation_create(&shape,&basis,error,sizeof(error));require(prep,error);
  ProjectorOwner projectors(openmx_core_hvna_projectors_create(prep,nullptr,&table,error,sizeof(error)),openmx_core_hvna_projectors_destroy);
  openmx_core_vna_preparation_destroy(prep);require(bool(projectors),error);
  WorkspaceOwner owner(openmx_core_hvna_workspace_create(projectors.get(),&work,error,sizeof(error)),openmx_core_hvna_workspace_destroy);
  require(bool(owner) && !table.groups && !table.components && !table.energies && !work.matrix && !work.orbital_capacity,
          "C API empty basis/projector views");
}

void c_api_test()
{
  Fixture f;ProjectorOwner projectors(f.create(),openmx_core_hvna_projectors_destroy);require(bool(projectors),f.error);
  require(f.table.species==2 && f.table.groups==4 && f.table.components==8,"C API projector dimensions");
  WorkspaceOwner work(openmx_core_hvna_workspace_create(projectors.get(),&f.work,f.error,sizeof(f.error)),openmx_core_hvna_workspace_destroy);
  require(bool(work) && f.work.orbital_capacity==7 && f.work.matrix,f.error);
  const auto original=f.table;const auto original_work=f.work;
  for (int invalid=0;invalid<4;++invalid) {
    if(invalid==0)f.channels[1][1]=nullptr;
    auto* bad=openmx_core_hvna_projectors_create(invalid==1?nullptr:f.preparation.get(),
        invalid==2?nullptr:f.energies,invalid==3?nullptr:&f.table,f.error,sizeof(f.error));
    require(!bad && f.error[0] && !std::memcmp(&f.table,&original,sizeof(original)),"invalid input changed projector views");
    f.channels[1][1]=f.values[1][1];
  }
  require(!openmx_core_hvna_workspace_create(nullptr,&f.work,f.error,sizeof(f.error)) && f.error[0],"null projectors accepted");
  require(!openmx_core_hvna_workspace_create(projectors.get(),nullptr,nullptr,0),"null workspace output accepted");
  require(!std::memcmp(&original_work,&f.work,sizeof(original_work)),"failure changed workspace views");
  openmx_core_hvna_projectors_destroy(nullptr);openmx_core_hvna_workspace_destroy(nullptr);
}

void allocation_test()
{
  Fixture f;
  openmx::test::fail_allocation_after(1000);
  ProjectorOwner projectors(f.create(),openmx_core_hvna_projectors_destroy);
  const long count=1000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(bool(projectors) && count>=5,"projector allocations not intercepted");const auto original=f.table;
  for (long failure=0;failure<count;++failure) {
    const auto before=openmx::test::live_allocations();openmx::test::fail_allocation_after(failure);
    auto* bad=f.create();openmx::test::fail_allocation_after(-1);
    if(bad)openmx_core_hvna_projectors_destroy(bad);
    require(!bad && f.error[0] && openmx::test::live_allocations()==before,"partial projector table leaked");
    require(!std::memcmp(&original,&f.table,sizeof(original)),"allocation failure published projector views");
  }
  openmx::test::fail_allocation_after(1000);
  WorkspaceOwner retained(openmx_core_hvna_workspace_create(projectors.get(),&f.work,f.error,sizeof(f.error)),openmx_core_hvna_workspace_destroy);
  const long work_count=1000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(bool(retained) && work_count>=2,"workspace allocations not intercepted");const auto work=f.work;f.work.matrix[0]=31;
  for (long failure=0;failure<work_count;++failure) {
    const auto before=openmx::test::live_allocations();openmx::test::fail_allocation_after(failure);
    auto* bad=openmx_core_hvna_workspace_create(projectors.get(),&f.work,f.error,sizeof(f.error));
    openmx::test::fail_allocation_after(-1);if(bad)openmx_core_hvna_workspace_destroy(bad);
    require(!bad && f.error[0] && openmx::test::live_allocations()==before,"partial contraction workspace leaked");
    require(f.work.matrix==work.matrix && f.work.matrix[0]==31,"failure changed retained workspace");
  }
}
} // namespace
int main(int argc,char** argv)
{
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="layout")layout_test();
    else if(name=="ownership")ownership_test();
    else if(name=="empty")empty_test();
    else if(name=="c-api")c_api_test();
    else if(name=="allocation")allocation_test();
    else throw std::runtime_error("unknown HVNA workspace test");
    std::cout<<"PASS HVNA workspace "<<name<<'\n';return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
