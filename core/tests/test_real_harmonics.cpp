#include "openmx_core_real_harmonics.h"
#include "openmx_core_atomic.h"
#include "openmx_core_native_resources.h"
#include "allocation_faults.hpp"
#include <array>
#include <climits>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context context() {
  char error[128];Context owner(openmx_core_create_host(error,sizeof error),openmx_core_destroy);
  require(bool(owner),error);return owner;
}
bool same(OpenMXRealHarmonicData a,OpenMXRealHarmonicData b) {
  return a.orders==b.orders && a.width==b.width && a.coefficients==b.coefficients;
}
void table(OpenMXRealHarmonicData data,int maximum) {
  require(data.orders==std::size_t(maximum+1) && data.width==std::size_t(2*maximum+1) && data.coefficients,"wrong conversion shape");
  for(std::size_t l=0;l<data.orders;++l)for(std::size_t row=0;row<=2*l;++row) {
    double norm=0;
    for(std::size_t col=0;col<=2*l;++col) {
      const auto x=openmx_real_harmonic_coefficient(&data,l,row,col);norm+=x.r*x.r+x.i*x.i;
    }
    require(std::abs(norm-1)<1e-14,"conversion not initialized");
  }
  require(data.coefficients[0].r==1. && data.coefficients[0].i==0.,"s conversion differs");
}
void contents() {
  auto first=context(),second=context();OpenMXRealHarmonicData a,b,c;char error[128];OpenMXAtomicViews atomic{};
  require(openmx_core_atomic_get(first.get(),&atomic,error,sizeof error) && !atomic.real_harmonics.coefficients,"absent owner exposed");
  const auto early=atomic;
  require(openmx_core_real_harmonics_initialize(first.get(),14,&a,error,sizeof error),error);
  require(openmx_core_real_harmonics_initialize(second.get(),4,&b,error,sizeof error),error);
  require(a.coefficients!=b.coefficients,"calculations share conversion storage");table(a,14);table(b,4);
  openmx::test::fail_allocation_after(0);
  const int get=openmx_core_real_harmonics_get(first.get(),&c,error,sizeof error);
  const int init=openmx_core_real_harmonics_initialize(first.get(),14,&c,error,sizeof error);
  const int bound=openmx_core_atomic_get(first.get(),&atomic,error,sizeof error);
  const auto remaining=openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(get && init && bound && remaining==0 && same(a,c) && same(a,atomic.real_harmonics),"reuse/atomic binding copied or allocated");
  require(!early.real_harmonics.coefficients,"old atomic descriptor refreshed implicitly");
  openmx_core_finish(first.get());table(b,4);
}
void invalid() {
  auto owner=context();char error[128];const dcomplex sentinel{17.,31.};const OpenMXRealHarmonicData saved{17,31,&sentinel};auto out=saved;
  require(!openmx_core_real_harmonics_get(owner.get(),&out,error,sizeof error) && same(out,saved) && *error,"uninitialized getter changed output");
  require(!openmx_core_real_harmonics_initialize(nullptr,4,&out,error,sizeof error) && same(out,saved),"null owner accepted");
  require(!openmx_core_real_harmonics_initialize(owner.get(),4,nullptr,error,sizeof error),"null output accepted");
  require(!openmx_core_real_harmonics_get(nullptr,&out,nullptr,0) && same(out,saved),"null getter owner accepted");
  require(!openmx_core_real_harmonics_get(owner.get(),nullptr,nullptr,0),"null getter output accepted");
  for(int maximum:{INT_MIN,-1,65,INT_MAX})
    require(!openmx_core_real_harmonics_initialize(owner.get(),maximum,&out,error,sizeof error) && same(out,saved),"invalid degree changed output");
  OpenMXRealHarmonicData active;
  require(openmx_core_real_harmonics_initialize(owner.get(),4,&active,error,sizeof error),error);
  char tiny[2]={'x','y'};
  require(!openmx_core_real_harmonics_initialize(owner.get(),5,&out,tiny,1) && same(out,saved) && !tiny[0] && tiny[1]=='y',"resize accepted or diagnostic overran");
  require(openmx_core_real_harmonics_get(owner.get(),&out,error,sizeof error) && same(out,active),"failed resize invalidated existing borrow");
  openmx_core_finish(owner.get());out=saved;
  require(!openmx_core_real_harmonics_get(owner.get(),&out,error,sizeof error) && same(out,saved),"finished getter exposed data");
  require(!openmx_core_real_harmonics_initialize(owner.get(),4,&out,error,sizeof error) && same(out,saved),"finished context reinitialized");
}
struct Drain {OpenMXCoreContext* context;OpenMXRealHarmonicData view;bool* called;bool* valid;};
void drain(OpenMXNativeResources*,void* pointer) {
  const auto& state=*static_cast<Drain*>(pointer);OpenMXRealHarmonicData view{};OpenMXAtomicViews atomic{};char error[128];
  *state.called=true;
  *state.valid=openmx_core_real_harmonics_get(state.context,&view,error,sizeof error) && same(view,state.view) &&
      openmx_core_atomic_get(state.context,&atomic,error,sizeof error) && same(view,atomic.real_harmonics) && view.coefficients[0].r==1.;
}
void lifetime() {
  for(bool finish:{false,true}) {
    const auto live=openmx::test::live_allocations();bool called=false,valid=false;
    {
      auto owner=context();OpenMXRealHarmonicData view{};char error[128];
      require(openmx_core_real_harmonics_initialize(owner.get(),14,&view,error,sizeof error),error);
      OpenMXNativeResources* resources=nullptr;void* pointer=nullptr;
      require(openmx_core_native_resources_state(owner.get(),"real harmonics drain",sizeof(Drain),drain,&resources,&pointer,error,sizeof error),error);
      *static_cast<Drain*>(pointer)={owner.get(),view,&called,&valid};
      if(finish){openmx_core_finish(owner.get());require(called && valid,"finish freed table before native drain");openmx_core_finish(owner.get());}
    }
    require(called && valid && openmx::test::live_allocations()==live,"owner leaked or expired before native drain");
  }
}
void allocation() {
  bool complete=false;int failures=0;
  for(int index=0;index<16;++index) {
    auto owner=context();char error[128];OpenMXRealHarmonicData out{};const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(index);
    const int ok=openmx_core_real_harmonics_initialize(owner.get(),14,&out,error,sizeof error);
    openmx::test::fail_allocation_after(-1);
    if(ok){complete=true;table(out,14);break;}
    ++failures;require(*error && !out.coefficients && !out.orders && openmx::test::live_allocations()==live,"failed initialization leaked or changed output");
    OpenMXAtomicViews atomic{};
    require(openmx_core_atomic_get(owner.get(),&atomic,error,sizeof error) && !atomic.real_harmonics.coefficients,"partial owner published through atomic view");
    require(openmx_core_real_harmonics_initialize(owner.get(),14,&out,error,sizeof error),error);table(out,14);
  }
  require(complete && failures>=2,"owner/data fault sweep incomplete");
}
void concurrent() {
  const auto live=openmx::test::live_allocations();std::array<std::thread,4> threads;std::array<int,4> ok{};
  for(int t=0;t<4;++t)threads[t]=std::thread([&,t]{
    try {
      auto owner=context();OpenMXRealHarmonicData view{};char error[128];
      require(openmx_core_real_harmonics_initialize(owner.get(),4+t,&view,error,sizeof error),error);
      for(int repeat=0;repeat<20;++repeat){OpenMXAtomicViews atomic{};require(openmx_core_atomic_get(owner.get(),&atomic,error,sizeof error),error);table(atomic.real_harmonics,4+t);}
      ok[t]=1;
    }catch(...){}
  });
  for(auto& thread:threads)thread.join();for(int value:ok)require(value==1,"concurrent owner calls interfere");
  require(openmx::test::live_allocations()==live,"concurrent contexts leaked");
}
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected real harmonic core test name");const std::string_view name=argv[1];
    if(name=="contents")contents();else if(name=="invalid")invalid();else if(name=="lifetime")lifetime();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();else throw std::invalid_argument("unknown real harmonic core test");
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
