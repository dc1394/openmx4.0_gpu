#include <openmx_core_mixing.h>
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
  char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  require(bool(c),error);return c;
}
OpenMXMixingHistory* history(OpenMXCoreContext* c) {
  char error[256];OpenMXMixingHistory* result=nullptr;
  require(openmx_core_mixing_history(c,&result,error,sizeof(error)),error);return result;
}
void zero(const OpenMXMixingHistory* h) {
  require(h && h->adaptive_max_weight==0 && h->consecutive==0,"fresh adaptive controller is not zero");
  for(double energy:h->energy)require(energy==0,"fresh calculation inherited energy history");
}
void seed(OpenMXMixingHistory* h,int value) {
  h->adaptive_max_weight=.125*value;h->consecutive=value;
  for(int i=0;i<5;++i)h->energy[i]=100*value+i*.25;
}
void check(const OpenMXMixingHistory* h,int value) {
  require(h->adaptive_max_weight==.125*value && h->consecutive==value,"adaptive controller crossed calculations");
  for(int i=0;i<5;++i)require(h->energy[i]==100*value+i*.25,"energy history crossed calculations");
}
void ownership() {
  auto a=calculation(),b=calculation();auto* x=history(a.get());auto* y=history(b.get());
  require(x!=y,"calculations share mixing history");zero(x);zero(y);
  for(int n=1;n<=100;++n) {seed(x,n);check(x,n);zero(y);seed(y,-n);check(x,n);check(y,-n);*y={};}
}
void reuse() {
  auto c=calculation();auto* h=history(c.get());seed(h,7);
  openmx::test::fail_allocation_after(0);
  for(int n=0;n<100;++n)require(history(c.get())==h,"binding replaced history storage");
  require(openmx::test::remaining_allocations()==0,"history lookup allocated");
  openmx::test::fail_allocation_after(-1);check(h,7);
}
void invalid() {
  auto c=calculation();auto* h=history(c.get());seed(h,3);auto* saved=h;char error[256];
  require(!openmx_core_mixing_history(nullptr,&h,error,sizeof(error)) && *error && h==saved,
      "null context changed the borrowed output");
  require(!openmx_core_mixing_history(c.get(),nullptr,error,sizeof(error)) && *error,"null output accepted");
  require(!openmx_core_mixing_history(nullptr,&h,nullptr,0) && h==saved,"missing diagnostic storage changed failure");
  char tiny='x';require(!openmx_core_mixing_history(c.get(),nullptr,&tiny,1) && tiny=='\0',"tiny diagnostic not terminated");
  check(h,3);require(history(c.get())==h,"invalid request damaged the owner");
}
void failures() {
  auto c=calculation(),other=calculation();auto* live=history(other.get());seed(live,5);
  openmx::test::fail_allocation_after(0);auto* fresh=history(c.get());
  require(openmx::test::remaining_allocations()==0,"fresh history binding allocated");
  openmx::test::fail_allocation_after(-1);zero(fresh);check(live,5);
  // The domain value participates in the context's existing RAII lifetime;
  // allocation failure while constructing that context must remain atomic.
  char error[256];openmx::test::fail_allocation_after(10000);
  auto* probe=openmx_core_create_host(error,sizeof(error));require(probe,error);
  const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  openmx_core_destroy(probe);require(count>=1,"faults did not reach shared core construction");
  for(long position=0;position<count;++position) {
    const auto before=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(position);
    auto* failed=openmx_core_create_host(error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    require(!failed && *error && openmx::test::live_allocations()==before,"failed context leaked a partial history");
    check(live,5);require(history(c.get())==fresh,"context failure affected a live history");
  }
}
void lifetime() {
  const auto initial=openmx::test::live_allocations();
  for(int n=1;n<=100;++n) {
    auto a=calculation(),b=calculation();seed(history(a.get()),n);seed(history(b.get()),-n);
    openmx::test::fail_allocation_after(0);openmx_core_native_resources_clear(a.get());
    openmx_core_finish(a.get());openmx_core_finish(a.get());openmx::test::fail_allocation_after(-1);
    zero(history(a.get()));check(history(b.get()),-n);
    try {seed(history(a.get()),n+1);throw n;}catch(int){}
  }
  require(openmx::test::live_allocations()==initial,"history survived calculation destruction");
}
void concurrent() {
  std::array<int,4> ok{};std::array<std::thread,4> threads;
  for(int t=0;t<4;++t)threads[t]=std::thread([&,t] {
    try {
      auto c=calculation();auto* h=history(c.get());
      for(int n=1;n<=100;++n) {seed(h,100*t+n);require(history(c.get())==h,"independent worker lost history");check(h,100*t+n);}
      ok[t]=1;
    }catch(...){}
  });
  for(auto& thread:threads)thread.join();for(int value:ok)require(value==1,"independent worker histories interfere");
}
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected a history test name");const std::string_view name=argv[1];
    if(name=="ownership")ownership();else if(name=="reuse")reuse();else if(name=="invalid")invalid();
    else if(name=="failures")failures();else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();
    else throw std::runtime_error("unknown mixing history test");
  }catch(const std::exception& e) {openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
