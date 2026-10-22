#include <openmx/core/force_recovery.hpp>
#include "allocation_faults.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
struct Probe { int calls=0,mode=0;const void *outer=nullptr; };
int vna(const OpenMXForceVnaRecoveryView* view,void* opaque) {
  auto& probe=*static_cast<Probe*>(opaque);++probe.calls;
  require(view->projectors && view->density && view->offset && view->density_offset,"missing VNA packet storage");
  require(view->projectors!=probe.outer,"nested VNA packet borrowed outer storage");
  view->offset[view->items]=view->values;view->density_offset[view->items]=view->density_values;
  for(size_t i=0;i<view->values;++i)view->projectors[i]=float(i)+.25f;
  for(size_t i=0;i<view->density_values;++i)view->density[i]=double(i)-.5;
  if(probe.mode==2)throw std::runtime_error("VNA callback exception");
  return probe.mode!=1;
}
int hnl(const OpenMXForceHnlRecoveryView* view,void* opaque) {
  auto& probe=*static_cast<Probe*>(opaque);++probe.calls;
  require(view->item && view->data && view->length && view->offset && view->length_offset,"missing HNL packet storage");
  require(view->data!=probe.outer,"nested HNL packet borrowed outer storage");
  view->offset[view->items]=view->values;view->length_offset[view->items]=view->lengths;
  for(size_t i=0;i<view->items;++i)view->item[i]={1,2,3,4,5,6};
  for(size_t i=0;i<view->values;++i)view->data[i]=double(i)+.125;
  for(size_t i=0;i<view->lengths;++i)view->length[i]=int(i);
  if(probe.mode==2)throw 17;
  return probe.mode!=1;
}
int run(int kind,Probe& probe,char* error,size_t bytes,size_t items=17,size_t values=113,size_t extra=29) {
  return kind?openmx_core_force_hnl_recovery_run(items,values,extra,hnl,&probe,error,bytes):
      openmx_core_force_vna_recovery_run(items,values,extra,vna,&probe,error,bytes);
}
void layouts() {
  for(size_t items:{size_t(0),size_t(1),size_t(256)})for(size_t values:{size_t(0),size_t(127)}){
    openmx::core::ForceVnaRecoveryPacket packet(items,values,values/2);auto view=packet.view();Probe probe;
    require(view.items==items && view.values==values && view.density_values==values/2,"VNA extents changed");
    require(vna(&view,&probe),"VNA fill failed");
    openmx::core::ForceVnaRecoveryPacket independent(items,values,values/2);
    auto other=independent.view();other.projectors[0]=91;
    require(other.projectors!=view.projectors && view.projectors[0]!=91,"independent VNA packets alias");
    openmx::core::ForceHnlRecoveryPacket nonlocal(items,values,values/3);auto n=nonlocal.view();
    require(n.items==items && n.values==values && n.lengths==values/3,"HNL extents changed");
    require(hnl(&n,&probe),"HNL fill failed");
    require(reinterpret_cast<uintptr_t>(n.item)%alignof(OpenMXForceHnlRecoveryItem)==0,"HNL item alignment differs");
  }
}
void lifetime() {
  for(int kind=0;kind<2;++kind)for(int mode=0;mode<3;++mode){
    Probe probe;probe.mode=mode;char error[256];const auto live=openmx::test::live_allocations();
    require(run(kind,probe,error,sizeof(error))==(mode==0),"packet callback result changed");
    require(probe.calls==1 && (mode==0?!*error:bool(*error)),"callback/error delivery changed");
    require(openmx::test::live_allocations()==live,"packet survived callback return or exception");
  }
}
void invalid() {
  for(int kind=0;kind<2;++kind){Probe probe;char error[256];
    for(auto shape:{std::array<size_t,3>{257,1,1},{std::numeric_limits<size_t>::max(),0,0},
                   {1,std::numeric_limits<size_t>::max(),1},{1,1,std::numeric_limits<size_t>::max()}})
      require(!run(kind,probe,error,sizeof(error),shape[0],shape[1],shape[2]) && *error,"invalid packet shape accepted");
    require(!probe.calls,"invalid packet entered native callback");
  }
  require(!openmx_core_force_vna_recovery_run(1,1,1,nullptr,nullptr,nullptr,0),"missing VNA callback accepted");
  require(!openmx_core_force_hnl_recovery_run(1,1,1,nullptr,nullptr,nullptr,0),"missing HNL callback accepted");
}
void allocation() {
  for(int kind=0;kind<2;++kind){
    Probe probe;char error[256];openmx::test::fail_allocation_after(10000);
    const int ok=run(kind,probe,error,sizeof(error));
    const auto count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    require(ok && count==(kind?5:4),"packet allocation coverage changed");
    for(long fail=0;fail<count;++fail){
      probe={};const auto live=openmx::test::live_allocations();openmx::test::fail_allocation_after(fail);
      const int accepted=run(kind,probe,error,sizeof(error));openmx::test::fail_allocation_after(-1);
      require(!accepted && !probe.calls && *error,"failed packet allocation entered dispatch");
      require(openmx::test::live_allocations()==live,"partial packet allocation leaked");
    }
  }
}
int nested_vna(const OpenMXForceVnaRecoveryView* view,void*) {
  view->projectors[0]=42;Probe probe;probe.outer=view->projectors;probe.mode=2;char error[256];
  require(!openmx_core_force_vna_recovery_run(1,2,3,vna,&probe,error,sizeof(error)),"nested exception escaped");
  require(view->projectors[0]==42,"nested call altered outer packet");return 1;
}
int nested_hnl(const OpenMXForceHnlRecoveryView* view,void*) {
  view->data[0]=37;Probe probe;probe.outer=view->data;probe.mode=1;char error[256];
  require(!openmx_core_force_hnl_recovery_run(1,2,3,hnl,&probe,error,sizeof(error)),"nested return escaped");
  require(view->data[0]==37,"nested call altered outer packet");return 1;
}
void nested() {
  char error[256];require(openmx_core_force_vna_recovery_run(1,2,3,nested_vna,nullptr,error,sizeof(error)),error);
  require(openmx_core_force_hnl_recovery_run(1,2,3,nested_hnl,nullptr,error,sizeof(error)),error);
}
void concurrent() {
  std::array<int,4> ok{};std::array<std::thread,4> threads;
  for(size_t t=0;t<threads.size();++t)threads[t]=std::thread([&,t]{
    for(int i=0;i<64;++i){Probe probe;char error[256];if(!run(int(t%2),probe,error,sizeof(error)))return;}
    ok[t]=1;
  });
  for(auto& thread:threads)thread.join();for(auto value:ok)require(value==1,"concurrent packets failed");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="layouts")layouts();else if(name=="lifetime")lifetime();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="nested")nested();else if(name=="concurrent")concurrent();
    else throw std::invalid_argument("unknown recovery test");
    std::cout<<"PASS Force recovery "<<name<<'\n';return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
