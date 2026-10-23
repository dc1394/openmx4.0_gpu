#include <openmx_core_energy.h>
#include <openmx_core_native_resources.h>
#include "allocation_faults.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation() {
  char error[160];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  require(bool(c),error);return c;
}
OpenMXThermodynamicEnergies* values(OpenMXCoreContext* c) {
  char error[160];OpenMXThermodynamicEnergies* out=nullptr;
  require(openmx_core_thermodynamic_energies(c,&out,error,sizeof(error)) && !*error,error);return out;
}
std::array<double*,3> fields(OpenMXThermodynamicEnergies* e) {
  return {&e->total,&e->ionic_kinetic,&e->pressure_volume};
}
std::array<double,3> snapshot(const OpenMXThermodynamicEnergies* e) {
  return {e->total,e->ionic_kinetic,e->pressure_volume};
}
void seed(OpenMXThermodynamicEnergies* e,int n) {
  int i=0;for(auto* value:fields(e))*value=100*n+(i++)+.125;
}
bool matches(const OpenMXThermodynamicEnergies* e,int n) {
  int i=0;for(double value:snapshot(e))if(value!=100*n+(i++)+.125)return false;
  return true;
}
void zero(const OpenMXThermodynamicEnergies* e) {
  for(double value:snapshot(e))require(value==0,"fresh calculation inherited thermodynamic energies");
}
void ownership() {
  auto a=calculation(),b=calculation();auto* x=values(a.get());auto* y=values(b.get());
  require(x!=y,"two calculations share thermodynamic energies");zero(x);zero(y);
  OpenMXBandEnergies* band=nullptr;char error[160];
  require(openmx_core_band_energies(a.get(),&band,error,sizeof(error)),error);band->band=-7.5;

  for(int i=1;i<=100;++i){seed(x,i);zero(y);seed(y,-i);require(matches(x,i) && matches(y,-i),"energy values crossed calculations");*y={};}
  require(band->band==-7.5,"thermodynamic writes changed band energy");
}
void reuse() {
  auto c=calculation();auto* e=values(c.get());seed(e,7);
  for(int i=0;i<100;++i)require(values(c.get())==e && matches(e,7),"binding replaced or reset live energies");
}
void invalid() {
  auto c=calculation();auto* e=values(c.get());auto* saved=e;seed(e,3);char error[160];
  require(!openmx_core_thermodynamic_energies(nullptr,&e,error,sizeof(error)) && *error && e==saved,"null calculation changed output");
  require(!openmx_core_thermodynamic_energies(c.get(),nullptr,error,sizeof(error)) && *error,"null output accepted");
  require(!openmx_core_thermodynamic_energies(nullptr,&e,nullptr,0) && e==saved,"missing diagnostic changed failure");
  char tiny='x';require(!openmx_core_thermodynamic_energies(c.get(),nullptr,&tiny,1) && tiny=='\0',"tiny diagnostic not terminated");
  require(matches(e,3),"invalid binding changed live energies");
}
void allocation() {
  auto a=calculation(),b=calculation();auto* live=values(b.get());seed(live,5);
  openmx::test::fail_allocation_after(0);auto* first=values(a.get());
  for(int i=0;i<100;++i)require(values(a.get())==first,"allocation-free lookup replaced energies");
  require(openmx::test::remaining_allocations()==0,"initial/repeated energy lookup allocated");
  openmx::test::fail_allocation_after(-1);zero(first);require(matches(live,5),"lookup failure damaged another calculation");
}
struct Borrower { OpenMXThermodynamicEnergies* energy;int expected;int* observed; };
void drain(OpenMXNativeResources*,void* opaque) {
  auto* state=static_cast<Borrower*>(opaque);
  *state->observed=matches(state->energy,state->expected)?1:-1;
}
void borrow(OpenMXCoreContext* c,int expected,int* observed) {
  char error[160];OpenMXNativeResources* owner=nullptr;void* opaque=nullptr;
  require(openmx_core_native_resources_state(c,"thermodynamic-energy-borrower",sizeof(Borrower),drain,&owner,&opaque,error,sizeof(error)),error);
  auto* state=static_cast<Borrower*>(opaque);state->energy=values(c);state->expected=expected;state->observed=observed;
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
      require(cleared==1,"clear/finish reset energies before draining native borrowers");
      zero(values(a.get()));require(matches(values(b.get()),-i),"clear/finish changed another calculation");
      // Rebinding starts a fresh set of components, independent of old borrowers.
      seed(values(a.get()),i+1);require(matches(values(a.get()),i+1),"finished calculation cannot be reused");
    }
    require(destroyed==1,"destruction invalidated energies before native borrowers");
  }
  require(openmx::test::live_allocations()==initial,"thermodynamic energies survived calculation destruction");
}
void concurrent() {
  std::array<int,4> ok{};std::array<std::thread,4> threads;
  for(int t=0;t<4;++t)threads[t]=std::thread([&,t] {
    try {auto c=calculation();auto* e=values(c.get());
      for(int i=1;i<=100;++i){seed(e,100*t+i);require(values(c.get())==e && matches(e,100*t+i),"worker energy state crossed calculations");}
      ok[t]=1;
    }catch(...){}
  });
  for(auto& thread:threads)thread.join();for(int value:ok)require(value==1,"independent energy owners interfere");
}
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected thermodynamic energy test name");const std::string_view name=argv[1];
    if(name=="ownership")ownership();else if(name=="reuse")reuse();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();
    else throw std::runtime_error("unknown thermodynamic energy test");
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}
