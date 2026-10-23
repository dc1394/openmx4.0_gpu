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
OpenMXChemicalPotentials* values(OpenMXCoreContext* c) {
  char error[160];OpenMXChemicalPotentials* out=nullptr;
  require(openmx_core_chemical_potentials(c,&out,error,sizeof(error)) && !*error,error);return out;
}
void seed(OpenMXChemicalPotentials* v,int n) { *v={100*n+.125,{100*n+.25,100*n+.5}}; }
bool matches(const OpenMXChemicalPotentials* v,int n) {
  return v->chemical_potential==100*n+.125 && v->xanes[0]==100*n+.25 && v->xanes[1]==100*n+.5;
}
void zero(const OpenMXChemicalPotentials* v) {
  require(v->chemical_potential==0 && v->xanes[0]==0 && v->xanes[1]==0,
          "fresh calculation inherited chemical potentials");
}
void ownership() {
  auto a=calculation(),b=calculation();auto* x=values(a.get());auto* y=values(b.get());
  require(x!=y,"two calculations share chemical potentials");zero(x);zero(y);
  char error[160];OpenMXBandEnergies* band=nullptr;OpenMXThermodynamicEnergies* physical=nullptr;
  OpenMXDynamicsState* dynamics=nullptr;
  require(openmx_core_band_energies(a.get(),&band,error,sizeof(error)),error);
  require(openmx_core_thermodynamic_energies(a.get(),&physical,error,sizeof(error)),error);
  require(openmx_core_dynamics(a.get(),&dynamics,error,sizeof(error)),error);
  band->band=-7.5;physical->total=-12.5;dynamics->temperature=300;
  for(int i=1;i<=100;++i) {
    seed(x,i);zero(y);seed(y,-i);
    require(matches(x,i) && matches(y,-i),"chemical potentials crossed calculations");*y={};
  }
  require(band->band==-7.5 && physical->total==-12.5 && dynamics->temperature==300,
          "chemical potentials changed energies or temperature");
}
void reuse() {
  auto c=calculation();auto* v=values(c.get());seed(v,7);
  for(int i=0;i<100;++i)require(values(c.get())==v && matches(v,7),"binding replaced or reset live chemical potentials");
}
void invalid() {
  auto c=calculation();auto* v=values(c.get());auto* saved=v;seed(v,3);char error[160];
  require(!openmx_core_chemical_potentials(nullptr,&v,error,sizeof(error)) && *error && v==saved,
          "null calculation changed output");
  require(!openmx_core_chemical_potentials(c.get(),nullptr,error,sizeof(error)) && *error,"null output accepted");
  require(!openmx_core_chemical_potentials(nullptr,&v,nullptr,0) && v==saved,"missing diagnostic changed failure");
  char tiny='x';require(!openmx_core_chemical_potentials(c.get(),nullptr,&tiny,1) && tiny=='\0',"tiny diagnostic not terminated");
  require(matches(v,3),"invalid binding changed live chemical potentials");
}
void allocation() {
  auto a=calculation(),b=calculation();auto* live=values(b.get());seed(live,5);
  openmx::test::fail_allocation_after(0);auto* first=values(a.get());
  for(int i=0;i<100;++i)require(values(a.get())==first,"allocation-free lookup replaced chemical potentials");
  require(openmx::test::remaining_allocations()==0,"initial/repeated chemical-potential lookup allocated");
  openmx::test::fail_allocation_after(-1);zero(first);require(matches(live,5),"lookup damaged another calculation");
}
struct Borrower { OpenMXChemicalPotentials* values;int expected;int* observed; };
void drain(OpenMXNativeResources*,void* opaque) {
  auto* state=static_cast<Borrower*>(opaque);*state->observed=matches(state->values,state->expected)?1:-1;
}
void borrow(OpenMXCoreContext* c,int expected,int* observed) {
  char error[160];OpenMXNativeResources* owner=nullptr;void* opaque=nullptr;
  require(openmx_core_native_resources_state(c,"chemical-borrower",sizeof(Borrower),drain,
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
      require(cleared==1,"clear/finish reset chemical potentials before native borrowers drained");
      zero(values(a.get()));require(matches(values(b.get()),-i),"clear/finish changed another calculation");
      seed(values(a.get()),i+1);require(matches(values(a.get()),i+1),"finished calculation cannot be reused");
    }
    require(destroyed==1,"destruction invalidated chemical potentials before native borrowers drained");
  }
  require(openmx::test::live_allocations()==initial,"chemical-potential state survived destruction");
}
void concurrent() {
  std::array<int,4> ok{};std::array<std::thread,4> workers;
  for(int t=0;t<4;++t)workers[t]=std::thread([&,t] {
    try {
      auto c=calculation();auto* v=values(c.get());
      for(int i=1;i<=100;++i) {seed(v,100*t+i);require(values(c.get())==v && matches(v,100*t+i),"worker chemical potentials crossed calculations");}
      ok[t]=1;
    }catch(...){}
  });
  for(auto& worker:workers)worker.join();for(int value:ok)require(value==1,"independent chemical-potential owners interfere");
}
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected chemical-potential test name");const std::string_view name=argv[1];
    if(name=="ownership")ownership();else if(name=="reuse")reuse();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();
    else throw std::runtime_error("unknown chemical-potential test");
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}
