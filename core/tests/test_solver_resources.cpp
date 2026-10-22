#include "openmx_core_solver_workspace.h"
#include "allocation_faults.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
struct Probe {
  std::array<int,16> events{};
  int count=0,mode=0,items=3;
  bool scratch_alive=true;
};
struct Object {Probe* probe;int id;const double* scratch;};
int drain(void* p,int device){auto& o=*static_cast<Object*>(p);o.probe->events[o.probe->count++]=10+device;return 0;}
int release(void* p,int device){
  auto* o=static_cast<Object*>(p);auto& probe=*o->probe;
  probe.scratch_alive&=o->id==device && o->scratch[0]==42;
  probe.events[probe.count++]=100+device;delete o;return 0;
}
OpenMXNativeResources* group(OpenMXSolverInvocation* scope,const char* name="native-test"){
  OpenMXNativeResources* out;char error[256];
  require(openmx_core_solver_resources(scope,name,&out,error,sizeof(error)),error);return out;
}
int populate(OpenMXSolverInvocation* scope,void* opaque,double* result){
  auto& probe=*static_cast<Probe*>(opaque);auto* owner=group(scope,"long-native-invocation-group-name");
  char error[256];const size_t shape[]{1,1,1,1};OpenMXSolverRealTensor scratch;
  require(openmx_core_solver_real(scope,0,"native-release-scratch",shape,&scratch,error,sizeof(error)),error);
  scratch.values[0]=42;
  for(int i=0;i<probe.items;++i){
    auto object=std::make_unique<Object>(Object{&probe,i,scratch.values});
    require(openmx_core_native_adopt(owner,1,object.get(),i,release,drain,error,sizeof(error)),error);
    (void)object.release();
  }
  void* host=openmx_core_native_malloc(owner,32,sizeof(double),1);require(host,"host registration failed");
  require(openmx_core_native_resource_count(owner)==size_t(probe.items+1),"native buffer ownership differs");
  if(probe.mode==1)return 0;
  if(probe.mode==2)throw std::runtime_error("injected native-stage exception");
  *result=19;return 1;
}
void check(const Probe& probe){
  require(probe.count==2*probe.items && probe.scratch_alive,"native resource leaked or outlived solver scratch");
  for(int i=0;i<probe.items;++i)require(probe.events[i]>=10 && probe.events[i]<100,"release preceded a drain");
  for(int i=probe.items;i<probe.count;++i)require(probe.events[i]>=100,"resource not released after draining");
}
void lifetime(){
  for(int mode=0;mode<3;++mode){
    Probe probe;probe.mode=mode;char error[256];double result=31;
    const auto live=openmx::test::live_allocations();
    const int ok=openmx_core_solver_run(2,populate,&probe,&result,error,sizeof(error));
    require(ok==(mode==0) && result==(mode==0?19:31),"native-stage callback result differs");
    check(probe);require(openmx::test::live_allocations()==live,"native-stage owner leaked");
  }
}
int nested(OpenMXSolverInvocation* scope,void* opaque,double* result){
  char error[256];void* state;auto* owner=group(scope);
  require(openmx_core_native_state(owner,"state",64,&state,error,sizeof(error)),error);
  require(state!=opaque && static_cast<int*>(state)[0]==0,"nested invocation inherited outer state");
  static_cast<int*>(state)[0]=7;*result=7;return 1;
}
int reuse_callback(OpenMXSolverInvocation* scope,void*,double* result){
  auto* owner=group(scope);char error[256];void* state;
  require(openmx_core_native_state(owner,"state",64,&state,error,sizeof(error)),error);
  require(reinterpret_cast<uintptr_t>(state)%alignof(std::max_align_t)==0,"native state alignment differs");
  for(int i=0;i<64;++i)require(static_cast<unsigned char*>(state)[i]==0,"new invocation retained native state");
  static_cast<int*>(state)[0]=41;
  openmx::test::fail_allocation_after(0);
  OpenMXNativeResources* same=nullptr;void* again=nullptr;
  const int group_ok=openmx_core_solver_resources(scope,"native-test",&same,error,sizeof(error));
  const int state_ok=openmx_core_native_state(owner,"state",64,&again,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(group_ok && state_ok && same==owner && again==state && static_cast<int*>(state)[0]==41,
      "native state reuse allocated, moved or cleared values");
  auto* other=group(scope,"other");require(other!=owner,"native group names alias");
  require(openmx_core_native_state(other,"state",64,&again,error,sizeof(error)) && again!=state,"native state groups alias");
  double output;require(openmx_core_solver_run(1,nested,state,&output,error,sizeof(error)) && output==7,error);
  require(static_cast<int*>(state)[0]==41,"nested native owner changed outer state");
  *result=19;return 1;
}
void reuse(){for(int i=0;i<3;++i){char error[256];double result;require(openmx_core_solver_run(1,reuse_callback,nullptr,&result,error,sizeof(error)),error);}}
int invalid_callback(OpenMXSolverInvocation* scope,void*,double* result){
  char error[256];auto* owner=group(scope);auto* out=owner;
  require(!openmx_core_solver_resources(nullptr,"x",&out,error,sizeof(error)) && out==owner,"null scope changed output");
  for(const char* name:{static_cast<const char*>(nullptr),""})
    require(!openmx_core_solver_resources(scope,name,&out,error,sizeof(error)) && out==owner,"invalid group name changed output");
  require(!openmx_core_solver_resources(scope,"x",nullptr,error,sizeof(error)),"null group output accepted");
  void* state;require(openmx_core_native_state(owner,"state",64,&state,error,sizeof(error)),error);void* saved=state;
  require(!openmx_core_native_state(owner,"state",65,&state,error,sizeof(error)) && state==saved,"changed native ABI size accepted");
  require(!openmx_core_native_state(owner,"large",SIZE_MAX,&state,error,sizeof(error)) && state==saved,"native state extent overflow accepted");
  *result=19;return 1;
}
void invalid(){char error[256];double result;require(openmx_core_solver_run(1,invalid_callback,nullptr,&result,error,sizeof(error)),error);}
void partial(){for(int items=0;items<=3;++items){Probe p;p.mode=1;p.items=items;char error[256];double result=31;
  require(!openmx_core_solver_run(1,populate,&p,&result,error,sizeof(error)) && result==31,"early stage return published output");check(p);}}
void allocation(){
  Probe probe;char error[256];double result=31;openmx::test::fail_allocation_after(10000);
  const int ok=openmx_core_solver_run(1,populate,&probe,&result,error,sizeof(error));
  const auto count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(ok && count>10,"native-stage allocation interception failed");check(probe);
  for(long failure=0;failure<count;++failure){
    Probe p;result=31;const auto live=openmx::test::live_allocations();openmx::test::fail_allocation_after(failure);
    const int passed=openmx_core_solver_run(1,populate,&p,&result,error,sizeof(error));openmx::test::fail_allocation_after(-1);
    require(!passed && result==31 && openmx::test::live_allocations()==live,"partial native owner leaked or published output");
    require(p.scratch_alive && p.count%2==0,"partial native owner outlived solver scratch");
    for(int i=0;i<p.count/2;++i)require(p.events[i]<100,"partial initialization released before draining");
  }
}
int concurrent_callback(OpenMXSolverInvocation* scope,void*,double* result){
  auto* owner=group(scope);std::array<int,4> passed{};std::array<std::thread,4> threads;
  for(int w=0;w<4;++w)threads[w]=std::thread([&,w]{for(int i=0;i<50;++i){
    auto* p=static_cast<int*>(openmx_core_native_malloc(owner,32,sizeof(int),1));if(!p)return;
    p[31]=w*100+i;auto* q=static_cast<int*>(openmx_core_native_realloc(owner,p,64*sizeof(int)));
    if(!q || q[31]!=w*100+i)return;openmx_core_native_free(owner,q);
  }passed[w]=1;});
  for(auto& t:threads)t.join();for(int ok:passed)require(ok,"concurrent native registration failed");
  require(openmx_core_native_resource_count(owner)==0,"concurrent native buffers leaked");*result=19;return 1;
}
void concurrent(){char error[256];double result;require(openmx_core_solver_run(4,concurrent_callback,nullptr,&result,error,sizeof(error)),error);}
}
int main(int argc,char** argv){
  try{const std::string name=argc==2?argv[1]:"";
    if(name=="lifetime")lifetime();else if(name=="reuse")reuse();else if(name=="invalid")invalid();
    else if(name=="partial")partial();else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();
    else throw std::runtime_error("unknown solver resource test");
    std::cout<<"PASS solver resources "<<name<<'\n';return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
