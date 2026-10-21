#include <openmx/core/electronic_matrices.hpp>
#include "openmx_core_electronic.h"
#include "allocation_faults.hpp"
#include <climits>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context context() {
  char error[256];Context result(openmx_core_create(error,sizeof(error)),openmx_core_destroy);
  require(bool(result),error);return result;
}
struct Fixture {
  int local[3]{0,3,1},first[7]{0,1,3,2,4,5,6},second[7]{0,3,1,2,4,5,6};
  int species[7]{0,0,1,1,0,0,1},basis[2]{4,2},contracted[2]{2,1},projectors[2]{3,1};
  int counts[7]{0,2,0,1,0,0,0},n1[3]{1,2,3},n2[1]{2},n3[2]{3,1},n4[1]{4},n5[1]{5},n6[1]{6};
  int* neighbors[7]{nullptr,n1,n2,n3,n4,n5,n6};
  OpenMXElectronicMatrixInput input{};OpenMXElectronicViews views{};char error[256]{};
  Fixture() {
    input.atoms=6;input.local_atoms=2;input.first_halo_atoms=input.second_halo_atoms=1;input.species=2;
    input.spin=3;input.spin_orbit=1;input.mixing=5;input.nonlocal_components=3;
    input.density_history=2;input.hamiltonian_history=3;
    input.scratch_orbitals=5;input.scratch_neighbors=4;input.scratch_projectors=7;
    input.contracted=input.core_hole=input.xmcd=input.interactions=input.lno=1;
    input.guiding_density=input.partial_density=1;
    input.atom_species=species;input.orbitals=basis;input.contracted_orbitals=contracted;
    input.projector_components=projectors;input.local_to_global=local;
    input.first_to_global=first;input.second_to_global=second;
    input.neighbors=counts;input.neighbor_atoms=neighbors;
  }
  int allocate(OpenMXCoreContext* c) { return openmx_core_electronic_allocate(c,&input,&views,error,sizeof(error)); }
};

// Checked against the pre-migration allocator with distinct M/F/S maps, halo
// derivatives, two scratch slots and non-collinear Hamiltonian histories.
constexpr std::size_t legacy_elements[]{180,48,135,195,51,135,195,180,180,135,135,999,
  1008,68,260,68,1608,1368,196,540,405,540,405,360,180,90,90,0,2,180,180,180};

void check_adapter(const OpenMXElectronicMatrixView& f,double***** p,std::size_t offset)
{
  for (std::size_t c=0;c<f.channels;++c) {
    const auto begin=f.component_offsets[offset+c],end=f.component_offsets[offset+c+1];
    for (auto a=begin;a<end;++a) {
      const auto first=f.atom_offsets[a],last=f.atom_offsets[a+1];
      for (auto b=first;b<last;++b) {
        const int rows=f.shapes[2*b],columns=f.shapes[2*b+1];
        require(f.block_offsets[b+1]-f.block_offsets[b]==static_cast<std::size_t>(rows)*columns,"block extent mismatch");
        for (int i=0;i<rows;++i) {
          const auto index=f.block_offsets[b]+static_cast<std::size_t>(i)*columns;
          require(openmx_electronic_row(&f,offset+c,a-begin,b-first,i)==p[c][a-begin][b-first][i],"CSR and legacy row differ");
          for (int j=0;j<columns;++j) p[c][a-begin][b-first][i][j]=static_cast<double>(1+index+j);
        }
      }
    }
  }
}
void layout_test()
{
  auto core=context();Fixture f;require(f.allocate(core.get()),f.error);
  const auto& v=f.views;
  for (std::size_t i=0;i<OPENMX_ELECTRONIC_FIELD_COUNT;++i) {
    const auto& field=v.fields[i];require(field.elements==legacy_elements[i],"legacy element count changed");
    for (std::size_t j=0;j<field.elements;++j) require(field.values[j]==0,"numeric storage is not initialized");
  }
  if (v.H0) check_adapter(v.fields[OPENMX_ELECTRONIC_KINETIC],v.H0,0);
  if (v.CntH0) check_adapter(v.fields[OPENMX_ELECTRONIC_CONTRACTED_KINETIC],v.CntH0,0);
  if (v.HNL) check_adapter(v.fields[OPENMX_ELECTRONIC_NONLOCAL],v.HNL,0);
  if (v.iHNL) check_adapter(v.fields[OPENMX_ELECTRONIC_IMAGINARY_NONLOCAL],v.iHNL,0);
  if (v.iCntHNL) check_adapter(v.fields[OPENMX_ELECTRONIC_CONTRACTED_IMAGINARY_NONLOCAL],v.iCntHNL,0);
  if (v.HCH) check_adapter(v.fields[OPENMX_ELECTRONIC_CORE_HOLE],v.HCH,0);
  if (v.iHCH) check_adapter(v.fields[OPENMX_ELECTRONIC_IMAGINARY_CORE_HOLE],v.iHCH,0);
  if (v.H_XMCD) check_adapter(v.fields[OPENMX_ELECTRONIC_XMCD],v.H_XMCD,0);
  if (v.H_Hub) check_adapter(v.fields[OPENMX_ELECTRONIC_HUBBARD],v.H_Hub,0);
  if (v.iHNL0) check_adapter(v.fields[OPENMX_ELECTRONIC_IMAGINARY_NONLOCAL_REFERENCE],v.iHNL0,0);
  if (v.OLP_L) check_adapter(v.fields[OPENMX_ELECTRONIC_ANGULAR_OVERLAP],v.OLP_L,0);
  if (v.OLP) check_adapter(v.fields[OPENMX_ELECTRONIC_OVERLAP],v.OLP,0);
  if (v.OLP_CH) check_adapter(v.fields[OPENMX_ELECTRONIC_CORE_HOLE_OVERLAP],v.OLP_CH,0);
  if (v.CntOLP) check_adapter(v.fields[OPENMX_ELECTRONIC_CONTRACTED_OVERLAP],v.CntOLP,0);
  if (v.H) check_adapter(v.fields[OPENMX_ELECTRONIC_HAMILTONIAN],v.H,0);
  if (v.CntH) check_adapter(v.fields[OPENMX_ELECTRONIC_CONTRACTED_HAMILTONIAN],v.CntH,0);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_PROJECTORS].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_PROJECTORS],v.DS_NL[g],g*v.fields[OPENMX_ELECTRONIC_PROJECTORS].channels);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_CONTRACTED_PROJECTORS].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_CONTRACTED_PROJECTORS],v.CntDS_NL[g],g*v.fields[OPENMX_ELECTRONIC_CONTRACTED_PROJECTORS].channels);
  if (v.DM0) check_adapter(v.fields[OPENMX_ELECTRONIC_LNO_DENSITY],v.DM0,0);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_HAMILTONIAN_HISTORY].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_HAMILTONIAN_HISTORY],v.HisH1[g],g*v.fields[OPENMX_ELECTRONIC_HAMILTONIAN_HISTORY].channels);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_IMAGINARY_HAMILTONIAN_HISTORY].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_IMAGINARY_HAMILTONIAN_HISTORY],v.HisH2[g],g*v.fields[OPENMX_ELECTRONIC_IMAGINARY_HAMILTONIAN_HISTORY].channels);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_HAMILTONIAN_RESIDUAL].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_HAMILTONIAN_RESIDUAL],v.ResidualH1[g],g*v.fields[OPENMX_ELECTRONIC_HAMILTONIAN_RESIDUAL].channels);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_IMAGINARY_HAMILTONIAN_RESIDUAL].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_IMAGINARY_HAMILTONIAN_RESIDUAL],v.ResidualH2[g],g*v.fields[OPENMX_ELECTRONIC_IMAGINARY_HAMILTONIAN_RESIDUAL].channels);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_DENSITY].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_DENSITY],v.DM[g],g*v.fields[OPENMX_ELECTRONIC_DENSITY].channels);
  if (v.ParDM) check_adapter(v.fields[OPENMX_ELECTRONIC_GUIDING_DENSITY],v.ParDM,0);
  if (v.iParDM) check_adapter(v.fields[OPENMX_ELECTRONIC_IMAGINARY_GUIDING_DENSITY],v.iParDM,0);
  if (v.Partial_DM) check_adapter(v.fields[OPENMX_ELECTRONIC_PARTIAL_DENSITY],v.Partial_DM,0);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_DENSITY_RESIDUAL].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_DENSITY_RESIDUAL],v.ResidualDM[g],g*v.fields[OPENMX_ELECTRONIC_DENSITY_RESIDUAL].channels);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_IMAGINARY_DENSITY_RESIDUAL].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_IMAGINARY_DENSITY_RESIDUAL],v.iResidualDM[g],g*v.fields[OPENMX_ELECTRONIC_IMAGINARY_DENSITY_RESIDUAL].channels);
  if (v.EDM) check_adapter(v.fields[OPENMX_ELECTRONIC_ENERGY_DENSITY],v.EDM,0);
  if (v.PDM) check_adapter(v.fields[OPENMX_ELECTRONIC_PREVIOUS_DENSITY],v.PDM,0);
  for (std::size_t g=0;g<v.fields[OPENMX_ELECTRONIC_IMAGINARY_DENSITY].groups;++g)
    check_adapter(v.fields[OPENMX_ELECTRONIC_IMAGINARY_DENSITY],v.iDM[g],g*v.fields[OPENMX_ELECTRONIC_IMAGINARY_DENSITY].channels);
  for (const auto& field:v.fields)
    for (std::size_t j=0;j<field.elements;++j) require(field.values[j]==static_cast<double>(j+1),"adapter overlaps or leaves holes");
}
void modes_test()
{
  auto core=context();
  for (int spin:{0,1,3}) for (int mixing=0;mixing<=7;++mixing) for (int flags:{0,1}) {
    Fixture f;f.input.spin=spin;f.input.mixing=mixing;
    f.input.contracted=f.input.core_hole=f.input.xmcd=f.input.interactions=f.input.lno=flags;
    f.input.spin_orbit=f.input.guiding_density=f.input.partial_density=flags;
    if (mixing!=1 && mixing!=5 && mixing!=6) f.input.hamiltonian_history=0;
    require(f.allocate(core.get()),f.error);const auto& v=f.views;
    require(bool(v.CntH)==bool(flags) && bool(v.iHNL)==(spin==3),"contraction/spin enablement");
    require(bool(v.iHCH)==(flags && spin==3) && bool(v.iCntHNL)==bool(flags),"imaginary core hole/contraction enablement");
    require(bool(v.HisH1)==(mixing==1 || mixing==5 || mixing==6),"Hamiltonian history enablement");
    require(bool(v.ResidualH1)==(mixing==5),"Hamiltonian residual enablement");
    require(v.fields[OPENMX_ELECTRONIC_DENSITY].elements==90*static_cast<std::size_t>(spin+1),"density history dimensions");
    require(v.fields[OPENMX_ELECTRONIC_PARTIAL_DENSITY].elements==(flags ? 90u : 0u),"partial density always has two channels");
    const bool dm=mixing==0 || mixing==1 || mixing==2 || mixing==6;
    require(bool(v.ResidualDM)==dm,"density residual enablement");
    require(v.fields[OPENMX_ELECTRONIC_IMAGINARY_DENSITY_RESIDUAL].elements==(dm && spin==3 && flags ? 180u : 2u),"imaginary residual dummy shape");
  }
  Fixture f;f.input.full_overlap_halo_derivatives=1;require(f.allocate(core.get()),f.error);
  require(f.views.fields[OPENMX_ELECTRONIC_OVERLAP].elements==1044,"full overlap halo derivatives");
  require(f.views.fields[OPENMX_ELECTRONIC_CORE_HOLE_OVERLAP].elements==1044,"full core overlap halo derivatives");
}
void mdspan_test()
{
  const std::size_t components[]{0,2,3},atoms[]{0,2,3,4};
  const MatrixBlockShape shapes[]{{2,3},{0,2},{3,0},{1,4}};
  RaggedMatrix matrix(components,atoms,shapes);
  require(matrix.elements()==10 && matrix.components()==2,"ragged component dimensions");
  auto b=matrix.block(1,0,0);element(b,0,3)=7;
  require(element(matrix.values(),9)==7 && b.mapping().stride(0)==4,"mdspan stride/offset");
  const auto& constant=matrix;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(element(constant.block(1,0,0),0,0))>>);
  require(element(constant.block(1,0,0),0,3)==7,"const mdspan view");
  require(matrix.block(0,0,1).extent(0)==0 && matrix.block(0,1,0).extent(1)==0,"zero extent block");
  for (int bad=0;bad<3;++bad) {
    bool caught=false;try { (void)matrix.block(bad==0 ? 2:1,bad==1 ? 1:0,bad==2 ? 1:0); }
    catch (const std::out_of_range&) { caught=true; }
    require(caught,"invalid block address accepted");
  }
  const std::size_t bad1[]{1,3},bad2[]{0,4},bad3[]{0,2,1,3};
  for (auto c:{std::span<const std::size_t>{},std::span<const std::size_t>(bad1),
      std::span<const std::size_t>(bad2),std::span<const std::size_t>(bad3)}) {
    bool caught=false;try { RaggedMatrix bad(c,atoms,shapes); }
    catch (const std::invalid_argument&) { caught=true; }
    require(caught,"invalid component CSR accepted");
  }
}
void ownership_test()
{
  auto a=context(),b=context();OpenMXElectronicViews kept{};
  {
    Fixture f;require(f.allocate(a.get()),f.error);kept=f.views;
    require(f.allocate(b.get()),f.error);require(kept.H!=f.views.H,"contexts share adapter");
    kept.H[0][1][0][0][0]=17;f.views.H[0][1][0][0][0]=41;
    f.local[1]=99;f.counts[3]=-1;f.basis[1]=-5;
  }
  b.reset();require(kept.H[0][1][0][0][0]==17,"ownership depends on input lifetime");
  OpenMXElectronicViews again{};char error[128];
  require(openmx_core_electronic_get(a.get(),&again,error,sizeof(error)) && again.H==kept.H,"lookup changes storage");
}
void lifecycle_test()
{
  auto c=context();OpenMXElectronicViews v{};char error[128];
  require(openmx_core_electronic_get(c.get(),&v,error,sizeof(error)) && !v.H,"initial owner not empty");
  Fixture f;
  for (int round=0;round<12;++round) {
    f.input.local_atoms=round%3;f.input.first_halo_atoms=f.input.second_halo_atoms=0;
    f.input.contracted=round%2;require(f.allocate(c.get()),f.error);
    require(f.views.fields[OPENMX_ELECTRONIC_KINETIC].atoms==4*static_cast<std::size_t>(f.input.local_atoms+1),"topology rebuild shape");
    require(f.views.H[0][0][0][0][0]==0,"replacement retains old numeric values");
    f.views.H[0][0][0][0][0]=9;
    if (round%3==0) openmx_core_electronic_release(c.get());
  }
  openmx_core_electronic_release(c.get());openmx_core_electronic_release(c.get());
  OpenMXElectronicMatrixInput empty{};empty.nonlocal_components=3;empty.density_history=empty.hamiltonian_history=1;empty.scratch_neighbors=1;
  require(openmx_core_electronic_allocate(c.get(),&empty,&v,error,sizeof(error)),error);
  require(v.fields[OPENMX_ELECTRONIC_KINETIC].elements==4 && v.fields[OPENMX_ELECTRONIC_PROJECTORS].elements==4,"empty rank dummy allocation");
  require(!v.DS_NL[0][0][1][0],"zero orbital scratch adapter");
  openmx_core_finish(c.get());openmx_core_finish(c.get());
  require(!f.allocate(c.get()) && !openmx_core_electronic_get(c.get(),&v,error,sizeof(error)),"finished context accepts operation");
}
void invalid_test()
{
  auto c=context();
  for (int kind=0;kind<26;++kind) {
    Fixture f;require(f.allocate(c.get()),f.error);const auto saved=f.views;saved.H[0][1][0][0][0]=19;
    switch(kind) {
      case 0:f.input.atoms=-1;break;case 1:f.input.local_atoms=7;break;
      case 2:f.input.first_halo_atoms=INT_MAX;break;case 3:f.input.second_halo_atoms=INT_MAX;break;
      case 4:f.input.spin=2;break;case 5:f.input.spin_orbit=-1;break;case 6:f.input.mixing=8;break;
      case 7:f.input.nonlocal_components=0;break;case 8:f.input.density_history=0;break;
      case 9:f.input.hamiltonian_history=-1;break;case 10:f.input.scratch_orbitals=3;break;
      case 11:f.input.scratch_projectors=4;break;case 12:f.input.scratch_neighbors=2;break;
      case 13:f.input.contracted=2;break;case 14:f.input.contracted_orbitals=nullptr;break;
      case 15:f.input.neighbor_atoms=nullptr;break;case 16:f.input.neighbors=nullptr;break;
      case 17:f.input.local_to_global=nullptr;break;case 18:f.local[1]=7;break;
      case 19:f.species[3]=2;break;case 20:f.n3[1]=7;break;case 21:f.counts[3]=-1;break;
      case 22:f.contracted[1]=3;break;case 23:f.projectors[1]=INT_MAX;break;
      case 24:f.input.atom_species=nullptr;break;case 25:f.input.orbitals=nullptr;break;
    }
    const auto live=openmx::test::live_allocations();
    require(!f.allocate(c.get()) && f.error[0],"invalid shape accepted");
    require(openmx::test::live_allocations()==live && !std::memcmp(&saved,&f.views,sizeof(saved)),"invalid replacement changes owner or leaks");
    OpenMXElectronicViews current{};
    require(openmx_core_electronic_get(c.get(),&current,f.error,sizeof(f.error)) && current.H==saved.H && current.H[0][1][0][0][0]==19,"failed replacement loses prior values");
  }
  Fixture f;
  require(!openmx_core_electronic_allocate(nullptr,&f.input,&f.views,nullptr,0),"null context accepted");
  require(!openmx_core_electronic_allocate(c.get(),nullptr,&f.views,nullptr,0),"null input accepted");
  require(!openmx_core_electronic_allocate(c.get(),&f.input,nullptr,nullptr,0),"null output accepted");
  char error[2]{'x','y'};
  require(!openmx_core_electronic_get(nullptr,&f.views,error,1) && error[0]==0 && error[1]=='y',"error buffer overrun");
}
void allocation_test()
{
  auto c=context();Fixture f;require(f.allocate(c.get()),f.error);
  openmx::test::fail_allocation_after(10000);const int ok=f.allocate(c.get());
  const auto count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(ok && count>100,"allocation interception unavailable");
  const auto saved=f.views;saved.H[0][1][0][0][0]=23;
  for (long fault=0;fault<count;++fault) {
    const auto live=openmx::test::live_allocations();openmx::test::fail_allocation_after(fault);
    const auto accepted=f.allocate(c.get());openmx::test::fail_allocation_after(-1);
    require(!accepted && f.error[0] && openmx::test::live_allocations()==live,"partial construction leaks");
    require(!std::memcmp(&saved,&f.views,sizeof(saved)) && saved.H[0][1][0][0][0]==23,"partial construction destroys old owner");
  }
}
void concurrent_test()
{
  int result[4]{};std::thread jobs[4];
  for (int t=0;t<4;++t) jobs[t]=std::thread([&,t] {
    try { auto c=context();Fixture f;
      for (int r=0;r<10;++r) {f.input.contracted=(r+t)%2;require(f.allocate(c.get()),f.error);
        f.views.H[0][1][0][0][0]=t+r;require(openmx_electronic_row(&f.views.fields[OPENMX_ELECTRONIC_HAMILTONIAN],0,1,0,0)[0]==t+r,"shared concurrent storage");}
      result[t]=1;
    } catch (...) {result[t]=-1;}
  });
  for (auto& job:jobs) job.join();for (int value:result) require(value==1,"independent concurrent contexts failed");
}
}
int main(int argc,char** argv)
{
  try {
    const std::string name=argc==2 ? argv[1] : "";
    if(name=="layout")layout_test();else if(name=="modes")modes_test();else if(name=="mdspan")mdspan_test();
    else if(name=="ownership")ownership_test();else if(name=="lifecycle")lifecycle_test();else if(name=="invalid")invalid_test();
    else if(name=="allocation")allocation_test();else if(name=="concurrent")concurrent_test();else throw std::runtime_error("unknown electronic matrix test");
    std::cout<<"PASS electronic matrices "<<name<<'\n';return 0;
  } catch (const std::exception& e) {openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
