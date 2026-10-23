#include <openmx_core_mixing.h>
#include "allocation_faults.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
void require(bool condition,const char* message)
{ if(!condition)throw std::runtime_error(message); }
void equal(double actual,double expected)
{ require(std::isfinite(actual) && std::abs(actual-expected)<1e-14*(1+std::abs(expected)),"unexpected adaptive weight"); }
using Snapshot=std::array<unsigned char,sizeof(OpenMXMixingHistory)>;
Snapshot snapshot(const OpenMXMixingHistory& h)
{ Snapshot bytes;std::memcpy(bytes.data(),&h,sizeof(h));return bytes; }
struct Primitives {
  int calls=0,signs=0,maxima=0,clamps=0,throw_at=-1;
  bool unknown_exception=false;
  void called() {
    if(calls++==throw_at) {
      if(unknown_exception)throw 7;
      throw std::runtime_error("native scalar failure");
    }
  }
  static double sign(void* context,double x) {
    auto& self=*static_cast<Primitives*>(context);++self.signs;self.called();
    return x<0?-1:1;
  }
  static double maximum(void* context,double a,double b) {
    auto& self=*static_cast<Primitives*>(context);++self.maxima;self.called();
    return b<=a?a:b;
  }
  static double clamp(void* context,double x,double minimum,double maximum,int* upper) {
    auto& self=*static_cast<Primitives*>(context);++self.clamps;self.called();
    *upper=!(x<maximum);
    return x<maximum?(minimum<x?x:minimum):maximum;
  }
  OpenMXMixingScalarOperations operations() {
    return {this,sign,maximum,clamp};
  }
};
OpenMXMixingHistory initial(bool falling=true,bool same_trend=true) {
  OpenMXMixingHistory h{};h.weight=.2;h.adaptive_max_weight=.4;h.consecutive=7;
  h.norm[0]=falling?1:4;h.norm[1]=falling?4:1;
  h.energy[0]=same_trend?2:0;h.energy[1]=1;h.energy[2]=0;
  h.norm[2]=19;h.norm[3]=23;h.norm[4]=29;h.energy[3]=31;h.energy[4]=37;
  return h;
}
void adapt(OpenMXMixingHistory& h,OpenMXMixingPolicy policy,Primitives& native,double low=.01,double high=.7) {
  char error[160]="old diagnostic";const auto operations=native.operations();
  require(openmx_core_mixing_adapt(&h,policy,low,high,&operations,error,sizeof(error)),error);
  require(!*error,"successful policy left a diagnostic");
}
void shift() {
  auto h=initial();h.norm[0]=-0.;h.energy[2]=std::numeric_limits<double>::quiet_NaN();
  const auto before=h;const auto weight=h.weight;const auto cap=h.adaptive_max_weight;
  require(openmx_core_mixing_shift(&h,nullptr,0),"valid history shift failed");
  for(int i=4;i>=1;--i) {
    require(!std::memcmp(h.norm+i,before.norm+i-1,sizeof(double)),"shift changed residual bits");
    require(!std::memcmp(h.energy+i,before.energy+i-1,sizeof(double)),"shift changed energy bits");
  }
  require(std::signbit(h.norm[0]) && h.energy[0]==before.energy[0],"shift overwrote current values");
  require(h.weight==weight && h.adaptive_max_weight==cap && h.consecutive==7,"shift changed adaptive control");
}
void ratio() {
  for(bool same:{false,true})for(bool falling:{false,true}) {
    auto h=initial(falling,same);Primitives native;const auto before=h;
    adapt(h,OPENMX_MIXING_DENSITY_RATIO,native);
    equal(h.weight,falling?4.0/15:.04);require(h.consecutive==(falling?7:-1),"ratio changed wrong counter branch");
    require(!std::memcmp(h.norm,before.norm,sizeof(h.norm)) && !std::memcmp(h.energy,before.energy,sizeof(h.energy)),"adapt shifted history");
    require(h.adaptive_max_weight==before.adaptive_max_weight,"adapt changed adaptive ceiling");
  }
  auto h=initial();Primitives native;adapt(h,OPENMX_MIXING_DENSITY_RATIO,native,.01,.25);
  equal(h.weight,.25);require(h.consecutive==8,"upper clamp did not increment counter");
  h=initial(false);adapt(h,OPENMX_MIXING_DENSITY_RATIO,native,.1,.7);equal(h.weight,.1);
  h=initial();h.norm[0]=h.norm[1]=1;const auto before=snapshot(h);adapt(h,OPENMX_MIXING_DENSITY_RATIO,native);
  require(snapshot(h)==before,"equal residuals changed weight or history");
  h=initial();h.norm[0]=0;h.norm[1]=1e-12;adapt(h,OPENMX_MIXING_DENSITY_RATIO,native,0,.7);equal(h.weight,.0002);
}
void potential() {
  for(bool same:{false,true})for(bool falling:{false,true}) {
    auto h=initial(falling,same);Primitives native;adapt(h,OPENMX_MIXING_KERKER_POTENTIAL_GAIN,native);
    const double expected=falling?(same?.32:.24):(same?.125:.1);
    equal(h.weight,expected);require(h.consecutive==(falling?7:-1),"potential counter branch changed");
    require(native.maxima==0 && native.clamps==1,"potential policy used density-ratio denominator");
  }
  auto h=initial();Primitives native;adapt(h,OPENMX_MIXING_KERKER_POTENTIAL_GAIN,native,.01,.3);
  equal(h.weight,.3);require(h.consecutive==8,"potential upper clamp did not increment counter");
}
void callbacks() {
  // Conditions short circuit exactly as in the native four-branch cascade.
  for(bool same:{true,false})for(bool falling:{true,false}) {
    auto h=initial(falling,same);Primitives native;
    adapt(h,OPENMX_MIXING_DENSITY_RATIO,native);
    require(native.signs==(same?(falling?2:4):(falling?6:8)),"energy primitive call sequence changed");
    require(native.maxima==1 && native.clamps==1,"ratio primitives were skipped or repeated");
  }
  auto a=initial(),b=initial(false);Primitives x,y;
  for(int i=0;i<50;++i) {adapt(a,OPENMX_MIXING_DENSITY_RATIO,x);adapt(b,OPENMX_MIXING_KERKER_POTENTIAL_GAIN,y);}
  require(x.maxima==50 && y.maxima==0 && x.clamps==50 && y.clamps==50,"native operation contexts crossed calculations");
}
void invalid() {
  auto h=initial();const auto before=snapshot(h);Primitives native;auto operations=native.operations();char error[160];
  require(!openmx_core_mixing_shift(nullptr,error,sizeof(error)) && *error,"null shift accepted");
  require(!openmx_core_mixing_adapt(nullptr,OPENMX_MIXING_DENSITY_RATIO,0,1,&operations,error,sizeof(error)) && *error,"null history accepted");
  require(!openmx_core_mixing_adapt(&h,OPENMX_MIXING_DENSITY_RATIO,0,1,nullptr,error,sizeof(error)) && *error,"null operations accepted");
  require(!openmx_core_mixing_adapt(&h,99,0,1,&operations,error,sizeof(error)) && *error,"invalid policy accepted");
  for(int i=0;i<3;++i) {
    operations=native.operations();
    if(i==0)operations.sign=nullptr;else if(i==1)operations.maximum=nullptr;else operations.clamp=nullptr;
    require(!openmx_core_mixing_adapt(&h,OPENMX_MIXING_DENSITY_RATIO,0,1,&operations,error,sizeof(error)) && *error,"missing scalar primitive accepted");
  }
  char tiny='x';require(!openmx_core_mixing_shift(nullptr,&tiny,1) && tiny=='\0',"tiny diagnostic unterminated");
  require(snapshot(h)==before && native.calls==0,"invalid call mutated state or called native code");
}
void failures() {
  for(bool unknown:{false,true})for(int position=0;position<10;++position) {
    auto h=initial(false,false);const auto before=snapshot(h);Primitives native;
    native.throw_at=position;native.unknown_exception=unknown;const auto operations=native.operations();char error[160];
    require(!openmx_core_mixing_adapt(&h,OPENMX_MIXING_DENSITY_RATIO,.01,.7,&operations,error,sizeof(error)) && *error,"throwing primitive escaped C ABI");
    require(snapshot(h)==before,"throwing primitive partially published state");
    native.throw_at=-1;adapt(h,OPENMX_MIXING_DENSITY_RATIO,native);equal(h.weight,.04);
  }
}
void allocation() {
  const auto before=openmx::test::live_allocations();auto a=initial(),b=initial(false);Primitives x,y;
  openmx::test::fail_allocation_after(0);
  for(int i=0;i<100;++i) {
    adapt(a,OPENMX_MIXING_DENSITY_RATIO,x);adapt(b,OPENMX_MIXING_KERKER_POTENTIAL_GAIN,y);
    require(openmx_core_mixing_shift(&a,nullptr,0) && openmx_core_mixing_shift(&b,nullptr,0),"allocation-free shift failed");
  }
  require(openmx::test::remaining_allocations()==0,"mixing policy allocated");
  openmx::test::fail_allocation_after(-1);
  require(openmx::test::live_allocations()==before,"mixing policy retained storage");
}
void concurrent() {
  std::array<int,4> ok{};std::array<std::thread,4> threads;
  for(int t=0;t<4;++t)threads[t]=std::thread([&,t] {
    try {Primitives native;
      for(int i=0;i<100;++i) {
        auto h=initial(t%2==0,t<2);adapt(h,OPENMX_MIXING_DENSITY_RATIO,native);
        equal(h.weight,t%2==0?4.0/15:.04);
      }
      require(native.clamps==100,"worker lost native operation context");ok[t]=1;
    }catch(...){}
  });
  for(auto& thread:threads)thread.join();for(int value:ok)require(value==1,"mixing policies shared worker state");
}
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected mixing policy test name");const std::string_view name=argv[1];
    if(name=="shift")shift();else if(name=="ratio")ratio();else if(name=="potential")potential();
    else if(name=="callbacks")callbacks();else if(name=="invalid")invalid();else if(name=="failures")failures();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();
    else throw std::runtime_error("unknown mixing policy test");
  }catch(const std::exception& exception) {
    openmx::test::fail_allocation_after(-1);std::cerr<<exception.what()<<'\n';return 1;
  }
}
