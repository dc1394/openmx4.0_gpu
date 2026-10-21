#include "openmx_core_native_resources.h"
#include "allocation_faults.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool ok,const char* text) { if(!ok)throw std::runtime_error(text); }
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context context() {
  char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  require(bool(c),error);return c;
}
OpenMXNativeResources* group(OpenMXCoreContext* c,const char* name="test",OpenMXNativeResourceReset reset=nullptr) {
  char error[256];OpenMXNativeResources* result=nullptr;
  require(openmx_core_native_resources(c,name,reset,&result,error,sizeof(error)),error);return result;
}
std::array<int,4096> events;
std::atomic<size_t> event_count{0};
void event(int code) { const auto n=event_count.fetch_add(1);if(n<events.size())events[n]=code; }
int released(void*,int device) { event(100+device);return 0; }
int failed(void*,int) { event(77);return 246; }
int sync_stream(void*,int device) { event(10+device);return 0; }
void reset() { event(1); }
void adopt(OpenMXNativeResources* g,int domain,void* pointer,int device,
    OpenMXNativeResourceRelease release=released,OpenMXNativeResourceRelease sync=nullptr) {
  char error[256];require(openmx_core_native_adopt(g,domain,pointer,device,release,sync,error,sizeof(error)),error);
}
void lifecycle() {
  int object;event_count=0;
  {
    auto c=context();auto* g=group(c.get(),"case",reset);
    require(g==group(c.get(),"case",reset),"group address changed");adopt(g,1,&object,3);
    openmx_core_finish(c.get());require(event_count==2 && events[0]==1 && events[1]==103,"finish release ordering");
    openmx_core_finish(c.get());require(event_count==2,"finish released twice");
    auto* fresh=group(c.get(),"case",reset);adopt(fresh,1,&object,5);
  }
  require(event_count==4 && events[2]==1 && events[3]==105,"context destruction did not own new group");
}
void host() {
  auto c=context();auto* g=group(c.get());
  auto* a=static_cast<unsigned char*>(openmx_core_native_malloc(g,16,7,1));require(a,"host allocation");
  for(int i=0;i<112;i++)require(a[i]==0,"calloc not zeroed");
  a[0]=7;a[111]=29;auto* grown=static_cast<unsigned char*>(openmx_core_native_realloc(g,a,8192));
  require(grown && grown[0]==7 && grown[111]==29,"realloc lost original values");
  require(openmx_core_native_resource_count(g)==1,"realloc duplicated ownership");
  require(!openmx_core_native_realloc(g,grown,SIZE_MAX) && grown[111]==29 &&
      openmx_core_native_resource_count(g)==1,"failed realloc lost previous storage");
  require(!openmx_core_native_malloc(g,SIZE_MAX,2,0),"multiplication overflow accepted");
  require(!openmx_core_native_realloc(g,grown,0) && !openmx_core_native_resource_count(g),"zero realloc did not release");
  auto* p=openmx_core_native_malloc(g,1,32,0);require(p,"host allocation");
  openmx_core_native_free(g,p);require(!openmx_core_native_resource_count(g),"host free retained ownership");
}
void domains() {
  auto c=context();auto* g=group(c.get());int object,status;event_count=0;
  adopt(g,1,&object,0);adopt(g,5,&object,2);
  require(openmx_core_native_resource_count(g)==2,"domains alias each other");
  require(openmx_core_native_release(g,5,&object,&status) && status==0,"handle release");
  require(!openmx_core_native_release(g,5,&object,&status),"handle was released twice");
  require(openmx_core_native_resource_count(g)==1,"handle freed device memory");
  openmx_core_native_resources_clear(c.get());require(event_count==2 && events[0]==102 && events[1]==100,"domain/device release");
}
void ordering() {
  auto c=context();auto* g=group(c.get());int memory,stream;event_count=0;
  adopt(g,6,&stream,2,released,sync_stream);adopt(g,1,&memory,1);
  openmx_core_native_resources_clear(c.get());
  require(event_count==3 && events[0]==12 && events[1]==101 && events[2]==102,"fallback freed buffers before draining streams");
}
void state() {
  auto c=context();auto* g=group(c.get());char error[256];void* pointer;
  require(openmx_core_native_state(g,"history",64,&pointer,error,sizeof(error)),error);
  require(reinterpret_cast<std::uintptr_t>(pointer)%alignof(std::max_align_t)==0,"state alignment");
  for(int i=0;i<64;i++)require(static_cast<unsigned char*>(pointer)[i]==0,"state not zeroed");
  static_cast<int*>(pointer)[3]=41;void* saved=pointer;
  openmx::test::fail_allocation_after(0);
  const int ok=openmx_core_native_state(g,"history",64,&pointer,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(ok && pointer==saved && static_cast<int*>(pointer)[3]==41,"state reuse allocated/reset values");
  require(!openmx_core_native_state(g,"history",65,&pointer,error,sizeof(error)),"state changed ABI size");
  require(pointer==saved,"state failure changed view");
  require(!openmx_core_native_state(g,"overflow",SIZE_MAX,&pointer,error,sizeof(error)) && pointer==saved,
      "oversized state changed output");
  openmx_core_native_resources_clear(c.get());g=group(c.get());
  require(openmx_core_native_state(g,"history",64,&pointer,error,sizeof(error)),error);
  require(static_cast<int*>(pointer)[3]==0,"new calculation inherited old history");
}
void invalid() {
  auto c=context();auto* g=group(c.get());char error[256];int object,status;event_count=0;
  adopt(g,1,&object,0,failed);
  require(!openmx_core_native_adopt(g,1,&object,0,released,nullptr,error,sizeof(error)),"duplicate adoption accepted");
  require(!openmx_core_native_adopt(g,0,&object,0,released,nullptr,error,sizeof(error)),"reserved domain accepted");
  require(openmx_core_native_release(g,1,&object,&status) && status==246,"SDK error was hidden");
  require(openmx_core_native_resource_count(g)==0 && event_count==1,"failed SDK release retained duplicate owner");
  auto* saved=g;
  require(!openmx_core_native_resources(c.get(),"test",reset,&g,error,sizeof(error)),"callback mismatch accepted");
  require(g==saved,"failed group lookup changed output");
}
void allocation() {
  auto c=context();int object;auto* g=group(c.get());char error[256];event_count=0;
  openmx::test::fail_allocation_after(0);
  int ok=openmx_core_native_adopt(g,1,&object,4,released,nullptr,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(!ok && !openmx_core_native_resource_count(g) && event_count==0,"failed adoption consumed caller's object");
  openmx::test::fail_allocation_after(0);
  void* host=openmx_core_native_malloc(g,1,64,0);
  openmx::test::fail_allocation_after(-1);
  require(!host && !openmx_core_native_resource_count(g),"failed host registration leaked ownership");
  // Every C++ allocation while constructing a group must leave native state
  // alone. Insertion failure must not run its not-yet-bound reset callback.
  for(long fail=0;fail<6;fail++) {
    auto separate=context();OpenMXNativeResources* output=nullptr;event_count=0;
    const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(fail);
    ok=openmx_core_native_resources(separate.get(),"long-new-native-group-name",reset,&output,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if(!ok)require(!output && event_count==0 && openmx::test::live_allocations()==live,"partial group creation leaked/reset native aliases");
  }
}
void concurrent() {
  auto c=context();auto* g=group(c.get());std::array<int,4> ok{};std::vector<std::thread> workers;event_count=0;
  for(int t=0;t<4;t++)workers.emplace_back([&,t] {
    for(int n=0;n<50;n++) {
      auto* p=static_cast<int*>(openmx_core_native_malloc(g,1,sizeof(int),0));if(!p)return;
      *p=t*100+n;auto* q=static_cast<int*>(openmx_core_native_realloc(g,p,32));if(!q)return;
      if(*q!=t*100+n)return;openmx_core_native_free(g,q);
      int object,status;char error[256];
      if(!openmx_core_native_adopt(g,1,&object,t,released,nullptr,error,sizeof(error)))return;
      if(!openmx_core_native_release(g,1,&object,&status) || status)return;
    }
    ok[t]=1;
  });
  for(auto& thread:workers)thread.join();for(auto value:ok)require(value==1,"concurrent ownership failed");
  require(!openmx_core_native_resource_count(g) && event_count==200,"worker resources leaked");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="lifecycle")lifecycle();else if(name=="host")host();else if(name=="domains")domains();
    else if(name=="ordering")ordering();else if(name=="state")state();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();else throw std::runtime_error("unknown resource test");
    std::cout<<"PASS native resources "<<name<<'\n';return 0;
  }catch(const std::exception& e) { openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1; }
}
