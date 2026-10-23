#include <openmx_core_cwf.h>
#include <openmx_core_chemical.h>
#include <openmx_core_energy.h>
#include <openmx_core_dynamics.h>
#include <openmx_core_native_resources.h>
#include "allocation_faults.hpp"
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation() {
  char error[160];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  require(bool(c),error);return c;
}
OpenMXCWFGrids* values(OpenMXCoreContext* c) {
  char error[160];OpenMXCWFGrids* out=nullptr;
  require(openmx_core_cwf_grids(c,&out,error,sizeof(error)) && !*error,error);return out;
}
constexpr std::array<int OpenMXCWFGrid::*,11> scalar_fields={
  &OpenMXCWFGrid::block_size,&OpenMXCWFGrid::process_rows,&OpenMXCWFGrid::process_columns,
  &OpenMXCWFGrid::local_rows,&OpenMXCWFGrid::local_columns,
  &OpenMXCWFGrid::maximum_rows,&OpenMXCWFGrid::maximum_columns,
  &OpenMXCWFGrid::process_row,&OpenMXCWFGrid::process_column,
  &OpenMXCWFGrid::system_handle,&OpenMXCWFGrid::context};
void seed(OpenMXCWFGrids* v,int n) {
  for(int i=0;i<OPENMX_CWF_GRID_COUNT;++i) {
    auto& g=v->matrix[i];const int base=1000*n+20*i;
    for(std::size_t j=0;j<scalar_fields.size();++j)g.*scalar_fields[j]=base+int(j)+1;
    for(int j=0;j<9;++j)g.descriptor[j]=base+12+j;
  }
}
bool matches(const OpenMXCWFGrids* v,int n) {
  for(int i=0;i<OPENMX_CWF_GRID_COUNT;++i) {
    const auto& g=v->matrix[i];const int base=1000*n+20*i;
    for(std::size_t j=0;j<scalar_fields.size();++j)if(g.*scalar_fields[j]!=base+int(j)+1)return false;
    for(int j=0;j<9;++j)if(g.descriptor[j]!=base+12+j)return false;
  }
  return true;
}
void zero(const OpenMXCWFGrids* v) {
  for(const auto& g:v->matrix) {
    for(auto field:scalar_fields)require(g.*field==0,"fresh calculation inherited CWF grid metadata");
    for(int value:g.descriptor)require(value==0,"fresh calculation inherited CWF descriptor");
  }
}
void ownership() {
  auto a=calculation(),b=calculation();auto* x=values(a.get());auto* y=values(b.get());
  require(x!=y,"two calculations share CWF grids");zero(x);zero(y);
  char error[160];OpenMXBandEnergies* band=nullptr;OpenMXThermodynamicEnergies* physical=nullptr;
  OpenMXDynamicsState* dynamics=nullptr;OpenMXChemicalPotentials* chemical=nullptr;
  require(openmx_core_band_energies(a.get(),&band,error,sizeof(error)),error);
  require(openmx_core_thermodynamic_energies(a.get(),&physical,error,sizeof(error)),error);
  require(openmx_core_dynamics(a.get(),&dynamics,error,sizeof(error)),error);
  require(openmx_core_chemical_potentials(a.get(),&chemical,error,sizeof(error)),error);
  chemical->chemical_potential=91.125;chemical->xanes[0]=7;chemical->xanes[1]=8;
  band->band=-7.5;physical->total=-12.5;dynamics->temperature=300;
  for(int i=1;i<=100;++i) {
    seed(x,i);zero(y);seed(y,-i);
    require(matches(x,i) && matches(y,-i),"CWF grids crossed calculations");*y={};
  }
  require(band->band==-7.5 && physical->total==-12.5 && dynamics->temperature==300 &&
          chemical->chemical_potential==91.125 && chemical->xanes[0]==7 && chemical->xanes[1]==8,
          "CWF grids changed energies or temperature");
}
void reuse() {
  auto c=calculation();auto* v=values(c.get());seed(v,7);
  for(int i=0;i<100;++i)require(values(c.get())==v && matches(v,7),"binding replaced or reset live CWF grids");
}
void invalid() {
  auto c=calculation();auto* v=values(c.get());auto* saved=v;seed(v,3);char error[160];
  require(!openmx_core_cwf_grids(nullptr,&v,error,sizeof(error)) && *error && v==saved,
          "null calculation changed output");
  require(!openmx_core_cwf_grids(c.get(),nullptr,error,sizeof(error)) && *error,"null output accepted");
  require(!openmx_core_cwf_grids(nullptr,&v,nullptr,0) && v==saved,"missing diagnostic changed failure");
  char tiny='x';require(!openmx_core_cwf_grids(c.get(),nullptr,&tiny,1) && tiny=='\0',"tiny diagnostic not terminated");
  require(matches(v,3),"invalid binding changed live CWF grids");
}
void allocation() {
  auto a=calculation(),b=calculation();auto* live=values(b.get());seed(live,5);
  openmx::test::fail_allocation_after(0);auto* first=values(a.get());
  for(int i=0;i<100;++i)require(values(a.get())==first,"allocation-free lookup replaced CWF grids");
  require(openmx::test::remaining_allocations()==0,"initial/repeated CWF-grid lookup allocated");
  openmx::test::fail_allocation_after(-1);zero(first);require(matches(live,5),"lookup damaged another calculation");
}
struct Borrower { OpenMXCWFGrids* values;int expected;int* observed; };
void drain(OpenMXNativeResources*,void* opaque) {
  auto* state=static_cast<Borrower*>(opaque);*state->observed=matches(state->values,state->expected)?1:-1;
}
void borrow(OpenMXCoreContext* c,int expected,int* observed) {
  char error[160];OpenMXNativeResources* owner=nullptr;void* opaque=nullptr;
  require(openmx_core_native_resources_state(c,"cwf-grid-borrower",sizeof(Borrower),drain,
          &owner,&opaque,error,sizeof(error)),error);
  auto* state=static_cast<Borrower*>(opaque);state->values=values(c);state->expected=expected;state->observed=observed;
}
void lifetime() {
  const auto initial=openmx::test::live_allocations();
  for(int action=0;action<2;++action)for(int i=1;i<=100;++i) {
    int cleared=0,destroyed=0;
    {
      auto a=calculation(),b=calculation();seed(values(a.get()),i);seed(values(b.get()),-i);
      borrow(a.get(),i,&cleared);borrow(b.get(),-i,&destroyed);
      openmx::test::fail_allocation_after(0);
      if(action==0)openmx_core_native_resources_clear(a.get());else openmx_core_finish(a.get());
      openmx_core_finish(a.get());openmx::test::fail_allocation_after(-1);
      require(cleared==1,"clear/finish reset CWF grids before native borrowers drained");
      zero(values(a.get()));require(matches(values(b.get()),-i),"clear/finish changed another calculation");
      seed(values(a.get()),i+1);require(matches(values(a.get()),i+1),"finished calculation cannot be reused");
    }
    require(destroyed==1,"destruction invalidated CWF grids before native borrowers drained");
  }
  require(openmx::test::live_allocations()==initial,"CWF-grid state survived destruction");
}
void concurrent() {
  std::array<int,4> ok{};std::array<std::thread,4> workers;
  for(int t=0;t<4;++t)workers[t]=std::thread([&,t] {
    try {
      auto c=calculation();auto* v=values(c.get());
      for(int i=1;i<=100;++i) {seed(v,100*t+i);require(values(c.get())==v && matches(v,100*t+i),"worker CWF grids crossed calculations");}
      ok[t]=1;
    }catch(...){}
  });
  for(auto& worker:workers)worker.join();for(int value:ok)require(value==1,"independent CWF-grid owners interfere");
}
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected CWF-grid test name");const std::string_view name=argv[1];
    if(name=="ownership")ownership();else if(name=="reuse")reuse();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();
    else throw std::runtime_error("unknown CWF-grid test");
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}
