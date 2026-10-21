#include "openmx_core_atomic.h"
#include "reciprocal_fixture.hpp"
#include "allocation_faults.hpp"
#include <cstring>
#include <iostream>
#include <string>

namespace {
using namespace openmx::test::reciprocal;
OpenMXAtomicViews atomic(OpenMXCoreContext *owner) {
  char error[256];OpenMXAtomicViews out{};
  require(openmx_core_atomic_get(owner,&out,error,sizeof(error)),error);return out;
}
void check2(OpenMXRadialView2 flat,double **rows,int species,int width) {
  require(flat.extent[0]==std::size_t(species) && flat.extent[1]==std::size_t(width),"radial capacity/padding");
  for (int s=0;s<species;++s) for (int r=0;r<width;++r) {
    auto *value=openmx_radial_row2(&flat,s)+r;
    require(value==&rows[s][r] && *value==0,"radial alias/initialization");
    *value=100*s+r+.5;require(rows[s][r]==100*s+r+.5,"radial view copies values");
  }
}
void layout_test() {
  auto owner=context();Fixture f;f.allocate_all(owner.get());auto v=atomic(owner.get());
  check2(v.pao_log_radii,f.pao.log_radii,3,5);check2(v.pao_radii,f.pao.radii,3,5);
  check2(v.density,f.pao.density,3,7);check2(v.density_backup,f.pao.density_backup,3,7);
  check2(v.vps_log_radii,f.vps.log_radii,3,5);check2(v.vps_radii,f.vps.radii,3,5);
  check2(v.neutral_potential,f.vps.neutral_potential,3,5);
  check2(v.hartree_potential,f.vps.hartree_potential,3,7);
  check2(v.partial_core_density,f.vps.partial_core_density,3,7);
  for (int s=0;s<3;++s) for (int l=0;l<3;++l) for (int m=0;m<2;++m) for (int r=0;r<5;++r)
    require(openmx_radial_row4(&v.orbitals,s,l,m)+r==&f.pao.orbitals[s][l][m][r],"PAO orbital stride");
  for (int spin=0;spin<2;++spin) for (int s=0;s<3;++s) for (int p=0;p<4;++p) for (int r=0;r<5;++r)
    require(openmx_radial_row4(&v.projectors,spin,s,p)+r==&f.vps.projectors[spin][s][p][r],"spin projector stride");
  require(v.projector_angular_species==3 && v.projector_angular_stride==4,"projector angular capacity");
  for (int s=0;s<3;++s) for (int p=0;p<4;++p) {
    auto *entry=openmx_atomic_projector_angular(&v,s)+p;
    require(entry==&f.vps.angular[s][p] && *entry==0,"projector angular alias/initialization");
    *entry=10*s+p;require(f.vps.angular[s][p]==10*s+p,"projector angular data copied");
  }
  for (int spin=0;spin<2;++spin) for (int s=0;s<3;++s) for (int p=0;p<4;++p) {
    f.vps.energies[spin][s][p]=100*spin+10*s+p;
    require(openmx_radial_row3(&v.projector_energies,spin,s)+p==&f.vps.energies[spin][s][p],"projector energy row alias");
    require(openmx_radial_value3(&v.projector_energies,spin,s,p)==100*spin+10*s+p,"projector energy stride");
  }
}
void lifecycle_test() {
  auto owner=context();Fixture f;
  require(!atomic(owner.get()).density.values && !atomic(owner.get()).species.species,"missing owners not empty");
  const OpenMXSpeciesShape shape{3,6,64};OpenMXSpeciesViews species{};
  require(openmx_core_species_allocate(owner.get(),&shape,&species,f.error,sizeof(f.error)),f.error);
  species.pao_mesh[0]=5;species.mass[2]=4;auto early=atomic(owner.get());
  require(early.species.pao_mesh==species.pao_mesh && !early.density.values,"early metadata view");
  f.allocate_all(owner.get());auto v=atomic(owner.get());
  require(!early.density.values && v.density.values,"allocation unexpectedly refreshes copied descriptor");
  v.density.values[0]=29;v.hartree_potential.values[0]=37;
  openmx_core_species_release(owner.get());
  require(!atomic(owner.get()).species.species && atomic(owner.get()).density.values[0]==29,"species release destroyed radial data");
  openmx_core_pao_release(owner.get());
  require(!atomic(owner.get()).density.values && atomic(owner.get()).hartree_potential.values[0]==37,"PAO release destroyed VPS");
  openmx_core_vps_release(owner.get());
  require(!atomic(owner.get()).vps_radii.values && !atomic(owner.get()).projectors.values && !atomic(owner.get()).projector_angular,"VPS release retains data");
  openmx_core_product_release(owner.get());
  f.allocate_all(owner.get());require(atomic(owner.get()).density.values[0]==0,"radial reallocation retains state");
  openmx_core_finish(owner.get());
}
void invalid_test() {
  auto owner=context();OpenMXAtomicViews out{},saved{};out.density.extent[0]=51;saved=out;char error[2]{'x','y'};
  require(!openmx_core_atomic_get(nullptr,&out,error,1) && !error[0] && error[1]=='y',"atomic error capacity");
  require(!std::memcmp(&out,&saved,sizeof(out)),"invalid getter changes output");
  require(!openmx_core_atomic_get(owner.get(),nullptr,nullptr,0),"null output accepted");
  openmx_core_finish(owner.get());
  require(!openmx_core_atomic_get(owner.get(),&out,nullptr,0) && !std::memcmp(&out,&saved,sizeof(out)),"finished getter");
}
void allocation_test() {
  auto owner=context();Fixture f;f.allocate_all(owner.get());auto before=atomic(owner.get());
  const auto live=openmx::test::live_allocations();
  openmx::test::fail_allocation_after(0);auto after=atomic(owner.get());
  const auto remaining=openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(!remaining && live==openmx::test::live_allocations() && before.density.values==after.density.values,
    "atomic getter allocates or replaces data");
}
void independent_test() {
  auto first=context(),second=context();Fixture a,b;a.allocate_all(first.get());b.allocate_all(second.get());
  auto x=atomic(first.get()),y=atomic(second.get());
  x.density.values[0]=23;y.density.values[0]=47;
  x.hartree_potential.values[0]=29;y.hartree_potential.values[0]=53;
  openmx_core_finish(first.get());
  require(atomic(second.get()).density.values[0]==47 && atomic(second.get()).hartree_potential.values[0]==53,
    "independent atomic ownership");
}
}
int main(int argc,char **argv) {
  try {
    const std::string name=argc>1 ? argv[1] : "";
    if (name=="layout") layout_test();else if (name=="lifecycle") lifecycle_test();
    else if (name=="invalid") invalid_test();else if (name=="allocation") allocation_test();
    else if (name=="independent") independent_test();else throw std::runtime_error("unknown atomic test");
    std::cout << "PASS atomic " << name << '\n';return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n';return 1; }
}
