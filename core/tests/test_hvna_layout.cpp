#include <openmx/core/hvna_layout.hpp>
#include "openmx_core_hvna.h"
#include "allocation_faults.hpp"
#include <array>
#include <climits>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool value,const char* reason) { if (!value) throw std::runtime_error(reason); }
using Owner=std::unique_ptr<OpenMXHvnaLayout,decltype(&openmx_core_hvna_destroy)>;
struct Fixture {
  int angular[2]{2,1}, first[3]{2,0,1}, second[2]{0,2};
  int* counts[2]{first,second};
  int local[3]{0,3,1}, extended[5]{0,3,1,4,2};
  int species[5]{0,1,0,0,1}, neighbors[5]{0,2,0,1,3};
  double energy[2][2][2]{{{1,-2},{3,4}},{{5,6},{-7,8}}};
  double* channels[2][2]{{energy[0][0],energy[0][1]},{energy[1][0],energy[1][1]}};
  double** energies[2]{channels[0],channels[1]};
  OpenMXHvnaInput input{4,2,2,local,extended,species,neighbors,energies};
  OpenMXVnaPreparationBuffers basis{};
  OpenMXHvnaBuffers views{};
  char error[256]{};
  std::unique_ptr<OpenMXVnaPreparation,decltype(&openmx_core_vna_preparation_destroy)> preparation{
      nullptr,&openmx_core_vna_preparation_destroy};
  Fixture() {
    const OpenMXVnaPreparationShape shape{2,2,1,2,3,2,angular,counts};
    preparation.reset(openmx_core_vna_preparation_create(&shape,&basis,error,sizeof(error)));
    require(bool(preparation),error);
  }
  OpenMXHvnaLayout* create() {
    return openmx_core_hvna_create(preparation.get(),&input,&views,error,sizeof(error));
  }
};

void layout_test()
{
  static_assert(!std::is_copy_constructible_v<HvnaLayout>);
  Fixture f;Owner owner(f.create(),openmx_core_hvna_destroy);require(bool(owner),f.error);
  // Global 3: seven orbitals, two neighbors; global 1: six orbitals, three neighbors.
  require(f.views.local_rows==5 && f.views.local_elements==256,"local projector archive shape");
  require(f.views.halo_rows==5 && f.views.halo_elements==248,"halo projector archive shape");
  const std::array<std::size_t,5> local{0,56,112,160,208},halo{0,48,96,144,192};
  for (std::size_t i=0;i<5;++i) {
    require(f.views.local_offsets[i]==local[i],"local block offset");
    require(f.views.halo_offsets[i]==halo[i],"halo block offset");
  }
  require(f.views.local_bases[1]==0 && f.views.local_bases[2]==2 && f.views.local_bases[3]==5 &&
          f.views.halo_bases[1]==0 && f.views.halo_bases[2]==4,"one-based atom bases");
  for (int s=0;s<2;++s) for (int l=0;l<2;++l) for (int r=0;r<2;++r) {
    require(f.views.energies[s*4+l*2+r]==f.energy[s][l][r],"energy species/projector order");
    require(f.views.lengths[l*2+r]==2*l+1,"projector magnetic length");
  }
}

void ownership_test()
{
  Owner retained(nullptr,openmx_core_hvna_destroy);OpenMXHvnaBuffers old{};
  for (int call=0;call<12;++call) {
    Fixture f;f.energy[0][0][0]=call;
    Owner next(f.create(),openmx_core_hvna_destroy);require(bool(next),f.error);
    f.preparation.reset();f.local[1]=4;f.energy[0][0][0]=-999;f.neighbors[3]=0;
    require(f.views.local_offsets[2]==112 && f.views.energies[0]==call,"layout borrowed input storage");
    if (retained) require(old.energies[0]==call-1 && old.energies!=f.views.energies,"layouts share storage");
    retained=std::move(next);old=f.views;
  }
  require(old.energies[0]==11 && old.halo_offsets[4]==192,"views depend on preparation lifetime");
}

void empty_test()
{
  Fixture f;f.input.atoms=f.input.local_atoms=f.input.halo_atoms=0;
  f.input.local_to_global=f.input.extended_to_global=f.input.species=f.input.neighbors=nullptr;
  Owner owner(f.create(),openmx_core_hvna_destroy);require(bool(owner),f.error);
  require(!f.views.local_rows && !f.views.halo_rows && !f.views.local_elements && !f.views.halo_elements,
          "empty rank has numeric rows");
  require(f.views.local_offsets && f.views.halo_offsets && f.views.local_bases && f.views.halo_bases,
          "empty archive lost dummy indexing storage");
  require(f.views.local_offsets[0]==0 && f.views.local_bases[1]==0,"empty archive sentinel");
  const std::array blocks{ProjectorAtomBlock{0,2},ProjectorAtomBlock{3,1}};
  ProjectorArchiveLayout zero(0,blocks);
  require(zero.elements()==0 && zero.rows()==3 && zero.offsets()[2]==0,"zero projector layout");
}

template<class Function> void throws(Function function) {
  bool rejected=false;
  try { function(); } catch(const std::invalid_argument&) { rejected=true; }
  catch(const std::length_error&) { rejected=true; }
  require(rejected,"invalid archive shape accepted");
}
void failure_test()
{
  throws([] { ProjectorArchiveLayout bad(-1,{}); });
  for (const auto block : {ProjectorAtomBlock{-1,1},ProjectorAtomBlock{1,0},ProjectorAtomBlock{1,-1}})
    throws([&] { ProjectorArchiveLayout bad(1,{&block,1}); });
  const std::array huge{ProjectorAtomBlock{0,INT_MAX},ProjectorAtomBlock{0,1}};
  throws([&] { ProjectorArchiveLayout bad(1,huge); });
  const std::array overflow{ProjectorAtomBlock{INT_MAX,INT_MAX}};
  throws([&] { ProjectorArchiveLayout bad(INT_MAX,overflow); });
  Fixture f;Owner owner(f.create(),openmx_core_hvna_destroy);require(bool(owner),f.error);
  const auto original=f.views;
  for (int invalid=0;invalid<10;++invalid) {
    const auto input=f.input;
    if (invalid==0)f.input.local_atoms=-1;
    if (invalid==1)f.input.halo_atoms=3;
    if (invalid==2)f.input.local_to_global=nullptr;
    if (invalid==3)f.input.extended_to_global=nullptr;
    if (invalid==4)f.local[1]=5;
    if (invalid==5)f.species[3]=2;
    if (invalid==6)f.neighbors[3]=INT_MAX;
    if (invalid==7)f.neighbors[3]=-1;
    if (invalid==8)f.channels[0][1]=nullptr;
    if (invalid==9)f.input.energies=nullptr;
    Owner bad(f.create(),openmx_core_hvna_destroy);
    require(!bad && f.error[0] && !std::memcmp(&original,&f.views,sizeof(original)),"failure published partial layout");
    f.input=input;f.local[1]=3;f.species[3]=0;f.neighbors[3]=1;f.channels[0][1]=f.energy[0][1];
  }
  require(!openmx_core_hvna_create(nullptr,&f.input,&f.views,f.error,sizeof(f.error)),"null preparation accepted");
  require(!openmx_core_hvna_create(f.preparation.get(),nullptr,&f.views,nullptr,0),"null input accepted");
  require(!openmx_core_hvna_create(f.preparation.get(),&f.input,nullptr,nullptr,0),"null output accepted");
  openmx_core_hvna_destroy(nullptr);
}

void allocation_test()
{
  Fixture f;
  openmx::test::fail_allocation_after(1000);
  Owner retained(f.create(),openmx_core_hvna_destroy);
  const long count=1000-openmx::test::remaining_allocations();
  openmx::test::fail_allocation_after(-1);
  require(bool(retained) && count>=7,"layout allocations not intercepted");
  const auto original=f.views;
  for (long failure=0;failure<count;++failure) {
    const auto before=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(failure);
    auto* bad=f.create();
    openmx::test::fail_allocation_after(-1);
    if (bad)openmx_core_hvna_destroy(bad);
    require(!bad && f.error[0] && openmx::test::live_allocations()==before,"partial HVNA layout leaked");
    require(!std::memcmp(&original,&f.views,sizeof(original)) && f.views.energies[0]==1,
            "allocation failure changed existing views");
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
    else if(name=="failure")failure_test();
    else if(name=="allocation")allocation_test();
    else throw std::runtime_error("unknown HVNA layout test");
    std::cout<<"PASS HVNA layout "<<name<<'\n';return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
