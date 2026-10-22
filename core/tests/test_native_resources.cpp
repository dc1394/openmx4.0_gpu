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
struct OwnedState {
  OpenMXNativeResources* owner;
  void* host;
  int* object;
  int* observed;
  int tag;
};
void owned_reset(OpenMXNativeResources* owner,void* metadata) {
  auto* state=static_cast<OwnedState*>(metadata);
  if(!state->tag)return; // A caller may unwind before initializing its POD.
  require(state->owner==owner,"reset used another calculation's owner");
  require(openmx_core_native_resource_count(owner)==3,"reset ran after resources were destroyed");
  int status=-1;
  require(openmx_core_native_release(owner,1,state->object,&status) && status==0,"reset could not release without deadlocking");
  require(*state->object==state->tag,"state/borrowed resource died before reset");
  openmx_core_native_free(owner,state->host);state->host=nullptr;
  *state->observed=state->tag;event(1000+state->tag);
}
OwnedState* owned_group(OpenMXCoreContext* c,const char* name="owned") {
  char error[256];OpenMXNativeResources* g=nullptr;void* state=nullptr;
  require(openmx_core_native_resources_state(c,name,sizeof(OwnedState),owned_reset,&g,&state,error,sizeof(error)),error);
  require(reinterpret_cast<std::uintptr_t>(state)%alignof(std::max_align_t)==0,"bound state alignment");
  auto* value=static_cast<OwnedState*>(state);
  if(!value->owner)value->owner=g;
  require(value->owner==g,"bound state owner changed");return value;
}
void populate(OwnedState* s,int* object,int* observed,int* stream) {
  s->tag=*object;s->object=object;s->observed=observed;
  s->host=openmx_core_native_malloc(s->owner,1,64,1);require(s->host,"owned state host storage");
  adopt(s->owner,1,object,*object);adopt(s->owner,6,stream,*stream,released,sync_stream);
}
void owned_lifecycle() {
  int a=1,b=2,d=3,sa=4,sb=5,sd=6,oa=0,ob=0,od=0;event_count=0;
  auto first=context();auto second=context();auto* x=owned_group(first.get());auto* y=owned_group(second.get());
  auto* z=owned_group(second.get(),"another");
  require(x!=y && y!=z && x->owner!=y->owner && y->owner!=z->owner,"contexts/names shared bound state");
  require(!x->tag && !y->host && !z->observed,"bound POD not zeroed");
  populate(x,&a,&oa,&sa);populate(y,&b,&ob,&sb);populate(z,&d,&od,&sd);
  char error[256];OpenMXNativeResources* g=nullptr;void* view=nullptr;
  openmx::test::fail_allocation_after(0);
  int ok=openmx_core_native_resources_state(second.get(),"owned",sizeof(OwnedState),owned_reset,&g,&view,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(ok && view==y && g==y->owner && y->tag==2,"SCF reuse allocated or reset state");
  event_count=0;openmx_core_native_resources_clear(first.get());
  require(oa==1 && !ob && !od && y->tag==2 && openmx_core_native_resource_count(y->owner)==3,"clearing first context touched second");
  require(event_count==4 && events[0]==101 && events[1]==1001 && events[2]==14 && events[3]==104,"state drain/fallback ordering");
  openmx_core_native_resources_clear(first.get());require(event_count==4,"clear drained twice");
  first.reset();require(event_count==4,"destroy drained cleared state again");
  openmx_core_finish(second.get());require(ob==2 && od==3 && event_count==12,"finish did not drain both exact states");
  openmx_core_finish(second.get());require(event_count==12,"finish drained twice");
  auto* fresh=owned_group(second.get());require(!fresh->tag && !fresh->host,"next calculation inherited stale metadata");
}
void empty_owned_reset(OpenMXNativeResources*,void*) { event(73); }
void owned_invalid() {
  auto c=context();auto* s=owned_group(c.get());char error[256];
  auto* owner=s->owner;void* metadata=s;
  for(auto bad_size:{size_t(0),sizeof(OwnedState)+1}) {
    require(!openmx_core_native_resources_state(c.get(),"owned",bad_size,owned_reset,&owner,&metadata,error,sizeof(error)),"state size mismatch accepted");
    require(owner==s->owner && metadata==s,"failure changed state/group output");
  }
  require(!openmx_core_native_resources_state(c.get(),"owned",sizeof(OwnedState),empty_owned_reset,&owner,&metadata,error,sizeof(error)),"state reset mismatch accepted");
  require(!openmx_core_native_resources(c.get(),"owned",nullptr,&owner,error,sizeof(error)),"legacy lookup erased state binding");
  (void)group(c.get(),"legacy");
  require(!openmx_core_native_resources_state(c.get(),"legacy",sizeof(OwnedState),owned_reset,&owner,&metadata,error,sizeof(error)),"state lookup changed legacy binding");
  require(!openmx_core_native_resources_state(c.get(),"overflow",SIZE_MAX,owned_reset,&owner,&metadata,error,sizeof(error)),"oversized bound state accepted");
  require(!openmx_core_native_resources_state(nullptr,"null",sizeof(OwnedState),owned_reset,&owner,&metadata,error,sizeof(error)),"missing calculation accepted");
  require(owner==s->owner && metadata==s,"invalid lookup published output");
}
void owned_allocation() {
  int failures=0,successes=0;
  for(long fail=0;fail<10;fail++) {
    auto c=context();char error[256];OpenMXNativeResources* owner=nullptr;void* metadata=nullptr;event_count=0;
    const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(fail);
    int ok=openmx_core_native_resources_state(c.get(),"long-bound-state-group-name",sizeof(OwnedState),empty_owned_reset,&owner,&metadata,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if(!ok) {
      require(!owner && !metadata && event_count==0 && openmx::test::live_allocations()==live,"partial bound state leaked or invoked unpublished reset");failures++;
    }else { require(owner && metadata && event_count==0,"creation called state reset");successes++; }
    openmx_core_native_resources_clear(c.get());require(event_count==(ok?1:0),"failed/successful insertion reset mismatch");
    require(openmx::test::live_allocations()==live,"bound state destruction retained metadata");
  }
  require(failures>=3 && successes>=1,"bound-state allocation fault sweep missed paths");
}
void owned_unwind() {
  int object=7,observed=0,stream=8;event_count=0;
  try {
    auto c=context();populate(owned_group(c.get()),&object,&observed,&stream);
    throw std::runtime_error("caller unwinds");
  }catch(const std::runtime_error&) {}
  require(observed==7,"exception did not drain bound state");
  require(event_count>=4 && events[event_count-3]==1007 && events[event_count-2]==18 && events[event_count-1]==108,"unwind destroyed state/resources out of order");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="lifecycle")lifecycle();else if(name=="host")host();else if(name=="domains")domains();
    else if(name=="ordering")ordering();else if(name=="state")state();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();
    else if(name=="owned-lifecycle")owned_lifecycle();else if(name=="owned-invalid")owned_invalid();
    else if(name=="owned-allocation")owned_allocation();else if(name=="owned-unwind")owned_unwind();else throw std::runtime_error("unknown resource test");
    std::cout<<"PASS native resources "<<name<<'\n';return 0;
  }catch(const std::exception& e) { openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1; }
}
