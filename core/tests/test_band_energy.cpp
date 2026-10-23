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
OpenMXBandEnergies* values(OpenMXCoreContext* c) {
  char error[160];OpenMXBandEnergies* out=nullptr;
  require(openmx_core_band_energies(c,&out,error,sizeof(error)) && !*error,error);return out;
}
void seed(OpenMXBandEnergies* e,int n) {
  e->band=100*n+.125;
  for(int spin=0;spin<2;++spin){e->os[spin]=10*n+spin+.25;e->is[spin]=-10*n-spin-.5;}
}
bool matches(const OpenMXBandEnergies* e,int n) {
  if(e->band!=100*n+.125)return false;
  for(int s=0;s<2;++s)if(e->os[s]!=10*n+s+.25 || e->is[s]!=-10*n-s-.5)return false;
  return true;
}
void zero(const OpenMXBandEnergies* e) {
  require(e->band==0 && e->os[0]==0 && e->os[1]==0 && e->is[0]==0 && e->is[1]==0,"fresh calculation inherited band energies");
}
void ownership() {
  auto a=calculation(),b=calculation();auto* x=values(a.get());auto* y=values(b.get());
  require(x!=y,"two calculations share electronic energies");zero(x);zero(y);
  for(int i=1;i<=100;++i){seed(x,i);zero(y);seed(y,-i);require(matches(x,i) && matches(y,-i),"energy values crossed calculations");*y={};}
}
void reuse() {
  auto c=calculation();auto* e=values(c.get());seed(e,7);
  for(int i=0;i<100;++i)require(values(c.get())==e && matches(e,7),"binding replaced or reset live energies");
}
void invalid() {
  auto c=calculation();auto* e=values(c.get());auto* saved=e;seed(e,3);char error[160];
  require(!openmx_core_band_energies(nullptr,&e,error,sizeof(error)) && *error && e==saved,"null calculation changed output");
  require(!openmx_core_band_energies(c.get(),nullptr,error,sizeof(error)) && *error,"null output accepted");
  require(!openmx_core_band_energies(nullptr,&e,nullptr,0) && e==saved,"missing diagnostic changed failure");
  char tiny='x';require(!openmx_core_band_energies(c.get(),nullptr,&tiny,1) && tiny=='\0',"tiny diagnostic not terminated");
  require(matches(e,3),"invalid binding changed live energies");
}
void allocation() {
  auto a=calculation(),b=calculation();auto* live=values(b.get());seed(live,5);
  openmx::test::fail_allocation_after(0);auto* first=values(a.get());
  for(int i=0;i<100;++i)require(values(a.get())==first,"allocation-free lookup replaced energies");
  require(openmx::test::remaining_allocations()==0,"initial/repeated energy lookup allocated");
  openmx::test::fail_allocation_after(-1);zero(first);require(matches(live,5),"lookup failure damaged another calculation");
}
struct Borrower { OpenMXBandEnergies* energy;int expected;int* observed; };
void drain(OpenMXNativeResources*,void* opaque) {
  auto* state=static_cast<Borrower*>(opaque);
  *state->observed=matches(state->energy,state->expected)?1:-1;
}
void borrow(OpenMXCoreContext* c,int expected,int* observed) {
  char error[160];OpenMXNativeResources* owner=nullptr;void* opaque=nullptr;
  require(openmx_core_native_resources_state(c,"energy-borrower",sizeof(Borrower),drain,&owner,&opaque,error,sizeof(error)),error);
  auto* state=static_cast<Borrower*>(opaque);state->energy=values(c);state->expected=expected;state->observed=observed;
}
void lifetime() {
  const auto initial=openmx::test::live_allocations();
  for(int i=1;i<=100;++i) {
    int cleared=0,destroyed=0;
    {
      auto a=calculation(),b=calculation();seed(values(a.get()),i);seed(values(b.get()),-i);
      borrow(a.get(),i,&cleared);borrow(b.get(),-i,&destroyed);
      openmx::test::fail_allocation_after(0);openmx_core_native_resources_clear(a.get());
      openmx_core_finish(a.get());openmx_core_finish(a.get());openmx::test::fail_allocation_after(-1);
      require(cleared==1,"clear reset energies before draining native borrowers");
      zero(values(a.get()));require(matches(values(b.get()),-i),"clear changed another calculation");
    }
    require(destroyed==1,"destruction invalidated energies before native borrowers");
  }
  require(openmx::test::live_allocations()==initial,"electronic energies survived calculation destruction");
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
    require(argc==2,"expected band energy test name");const std::string_view name=argv[1];
    if(name=="ownership")ownership();else if(name=="reuse")reuse();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();
    else throw std::runtime_error("unknown band energy test");
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}
