#include <openmx/core/species_storage.hpp>
#include "openmx_core_species.h"
#include "reciprocal_fixture.hpp"
#include "allocation_faults.hpp"
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

namespace {
using namespace openmx::core;
using openmx::test::reciprocal::require;
using openmx::test::reciprocal::context;
constexpr OpenMXSpeciesShape shape{3,6,64};
OpenMXSpeciesViews allocate(OpenMXCoreContext *owner) {
  char error[256];OpenMXSpeciesViews v{};
  require(openmx_core_species_allocate(owner,&shape,&v,error,sizeof(error)),error);
  return v;
}
OpenMXSpeciesViews get(OpenMXCoreContext *owner) {
  char error[256];OpenMXSpeciesViews v{};
  require(openmx_core_species_get(owner,&v,error,sizeof(error)),error);return v;
}
void layout_test() {
  auto owner=context();auto v=allocate(owner.get());
  require(v.species==3 && v.angular_channels==6 && v.text_capacity==64,"species dimensions");
  int *fields[]{v.basis_lmax,v.open_core,v.species_index,v.relativistic,v.vps_mesh,v.pao_mesh,
    v.projector_components,v.radial_projectors,v.pao_lmax,v.pao_multiplicity,
    v.atomic_number,v.orbitals,v.contracted_orbitals};
  for (int f=0;f<13;++f) for (int s=0;s<3;++s) {
    require(fields[f][s]==0,"metadata not initialized");fields[f][s]=100*f+s;
  }
  for (int f=0;f<13;++f) for (int s=0;s<3;++s)
    require(fields[f][s]==100*f+s,"metadata fields overlap");
  double *real[]{v.mass,v.electrons,v.cutoff,v.core_charge};
  for (int f=0;f<4;++f) for (int s=0;s<3;++s) {
    require(real[f][s]==(f==0 ? -1 : 0),"metadata real initialization");
    real[f][s]=10*f+s+.25;
  }
  for (int f=0;f<4;++f) for (int s=0;s<3;++s)
    require(real[f][s]==10*f+s+.25,"real fields overlap");
  char **labels[]{v.name,v.basis_name,v.basis_file,v.potential_file};
  for (int f=0;f<4;++f) for (int s=0;s<3;++s) {
    for (int c=0;c<64;++c) require(labels[f][s][c]==0,"label bytes not initialized");
    std::memset(labels[f][s],'a'+f*3+s,63);
    require(std::strlen(labels[f][s])==63,"label capacity/terminator");
  }
  for (int f=0;f<4;++f) for (int s=0;s<3;++s)
    require(labels[f][s][62]=='a'+f*3+s,"label fields overlap");
  for (int s=0;s<3;++s) for (int l=0;l<6;++l) {
    require(&v.basis_rows[s][l]==v.basis+s*6+l &&
      &v.contracted_basis_rows[s][l]==v.contracted_basis+s*6+l,"basis adapter copies numeric storage");
    require(v.basis_rows[s][l]==0 && v.contracted_basis_rows[s][l]==0,"basis initialization");
    v.basis[s*6+l]=10*s+l;v.contracted_basis_rows[s][l]=20*s+l;
    require(v.basis_rows[s][l]==10*s+l && v.contracted_basis[s*6+l]==20*s+l,"flat basis mapping");
  }
}
void mdspan_test() {
  SpeciesStorage v({3,6,64});
  auto b=v.basis(),c=v.basis(true);auto label=v.labels(SpeciesLabel::name);
  for (int s=0;s<3;++s) for (int l=0;l<6;++l) {
    element(b,s,l)=s*6+l;element(c,s,l)=100+s*6+l;
    require(&element(b,s,l)==b.data_handle()+s*6+l,"mdspan basis strides");
  }
  element(label,2,63)='x';
  require(label.extent(1)==64 && label.data_handle()[191]=='x',"mdspan label capacity");
  require(v.integers(SpeciesInteger::orbitals).size()==3 &&
    v.reals(SpeciesReal::mass)[2]==-1,"SoA field spans");
}
void lifecycle_test() {
  auto a=context(),b=context();auto first=allocate(a.get());auto second=allocate(b.get());
  first.mass[1]=41;second.mass[1]=17;first.basis_rows[1][3]=2;
  require(get(a.get()).mass[1]==41 && get(b.get()).mass[1]==17,"shared species owner");
  const auto live=openmx::test::live_allocations();
  openmx::test::fail_allocation_after(0);auto same=get(a.get());
  const auto remaining=openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(!remaining && openmx::test::live_allocations()==live && same.basis_rows==first.basis_rows,
    "species getter allocates or changes addresses");
  for (int i=0;i<8;++i) {
    openmx_core_species_release(a.get());openmx_core_species_release(a.get());
    require(!get(a.get()).species && !get(a.get()).name,"released species still visible");
    first=allocate(a.get());require(first.mass[1]==-1 && !first.basis_rows[1][3],"stale species on reuse");
  }
  openmx_core_finish(a.get());openmx_core_finish(a.get());
  require(get(b.get()).mass[1]==17,"finish affected independent calculation");
}
void invalid_test() {
  auto owner=context();char error[4];OpenMXSpeciesViews out{},saved{};
  out.species=47;saved=out;
  for (OpenMXSpeciesShape s : {OpenMXSpeciesShape{0,6,64},{3,0,64},{3,31,64},{3,6,1},{-1,6,64}}) {
    require(!openmx_core_species_allocate(owner.get(),&s,&out,error,sizeof(error)) && error[0],"invalid shape accepted");
    require(!std::memcmp(&out,&saved,sizeof(out)) && !get(owner.get()).species,"failure publishes state");
  }
  require(!openmx_core_species_allocate(nullptr,&shape,&out,nullptr,0),"null context allocation");
  require(!openmx_core_species_allocate(owner.get(),nullptr,&out,nullptr,0),"null shape");
  require(!openmx_core_species_allocate(owner.get(),&shape,nullptr,nullptr,0),"null allocation output");
  auto v=allocate(owner.get());v.mass[0]=13;
  require(!openmx_core_species_allocate(owner.get(),&shape,&out,nullptr,0) && get(owner.get()).mass[0]==13,
    "live owner replaced");
  require(!openmx_core_species_get(nullptr,&out,error,1) && !error[0],"getter error capacity");
  require(!openmx_core_species_get(owner.get(),nullptr,nullptr,0),"null getter output");
  openmx_core_finish(owner.get());
  require(!openmx_core_species_get(owner.get(),&out,nullptr,0) && !std::memcmp(&out,&saved,sizeof(out)),"finished getter");
  require(!openmx_core_species_allocate(owner.get(),&shape,&out,nullptr,0),"finished allocation");
}
void overflow_test() {
  bool rejected=false;
  try { SpeciesStorage huge({std::numeric_limits<std::size_t>::max(),6,64}); }
  catch (const std::length_error&) { rejected=true; }
  require(rejected,"species multiplication overflow");
}
void allocation_test() {
  long count=0;
  {
    auto owner=context();OpenMXSpeciesViews v{};char error[256];
    openmx::test::fail_allocation_after(10000);
    const int ok=openmx_core_species_allocate(owner.get(),&shape,&v,error,sizeof(error));
    count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    require(ok && count>0,"allocation count probe");
  }
  for (long fail=0;fail<count;++fail) {
    auto owner=context();OpenMXSpeciesViews v{},saved{};v.species=53;saved=v;char error[256];
    const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(fail);
    const int ok=openmx_core_species_allocate(owner.get(),&shape,&v,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    require(!ok && error[0] && !std::memcmp(&v,&saved,sizeof(v)) &&
      !get(owner.get()).species && openmx::test::live_allocations()==live,"allocation rollback leak/publication");
    require(allocate(owner.get()).mass[0]==-1,"failed owner cannot be reused");
  }
}
void concurrent_test() {
  std::vector<std::thread> threads;
  for (int worker=0;worker<4;++worker) threads.emplace_back([worker] {
    for (int round=0;round<8;++round) {
      auto owner=context();auto v=allocate(owner.get());v.mass[2]=worker*8+round;
      v.name[1][0]='a'+worker;require(get(owner.get()).mass[2]==worker*8+round,"concurrent metadata");
    }
  });
  for (auto& t:threads)t.join();
}
}
int main(int argc,char **argv) {
  try {
    const std::string name=argc>1 ? argv[1] : "";
    if (name=="layout") layout_test();else if (name=="mdspan") mdspan_test();
    else if (name=="lifecycle") lifecycle_test();else if (name=="invalid") invalid_test();
    else if (name=="overflow") overflow_test();else if (name=="allocation") allocation_test();
    else if (name=="concurrent") concurrent_test();else throw std::runtime_error("unknown species test");
    std::cout << "PASS species " << name << '\n';return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n';return 1; }
}
