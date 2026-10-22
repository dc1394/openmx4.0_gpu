#include <openmx/core/force_hnl_batch.hpp>
#include "allocation_faults.hpp"
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

namespace {
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
struct Probe {
  int entered=0,finished=0,mode=0;
  double *force=nullptr;
  const void *outer=nullptr;
};
int operate(const OpenMXForceHnlBatchView* view,void* arguments) {
  auto& p=*static_cast<Probe*>(arguments);++p.entered;
  require(view->item && view->atom && view->q && view->a && view->b && view->row_kind &&
      view->energy && view->projectors && view->density && view->force,"missing typed batch storage");
  require(bool(view->h)==(view->kind==2),"case-specific centre index changed");
  require(view->force!=p.outer,"nested batch aliases its caller");
  p.force=view->force;view->force[0]=17;
  for(size_t i=0;i<view->items;++i){
    view->item[i]={i*6,int(i),1,2,3};view->atom[i]=int(i)+1;view->q[i]=int(i)+2;
    if(view->h)view->h[i]=int(i)+3;
    for(int d=0;d<3;++d)view->force[3*i+d]=double(i)+.125*d;
  }
  for(size_t i=0;i<view->rows;++i){
    view->a[i]=i*11;view->b[i]=i*13;view->row_kind[i]=int(i%3);
    view->energy[i]=int(i%7);view->projectors[i]=int(i%5);
  }
  for(size_t i=0;i<view->density_values;++i)view->density[i]=double(i)-.25;
  if(p.mode==2)throw std::runtime_error("native dispatch failed");
  if(p.mode==3)throw 37;
  return p.mode!=1;
}
void finish(void* arguments) {
  auto& p=*static_cast<Probe*>(arguments);++p.finished;
  // ASan verifies that final device cleanup can still read/write the host view.
  p.force[0]=41;
}
int run(int kind,Probe& p,char* error,size_t bytes,size_t items=17,size_t rows=29,size_t density=113) {
  return openmx_core_force_hnl_batch_run(kind,items,kind==2?items:rows,density,operate,finish,&p,error,bytes);
}
void layouts() {
  static_assert(!std::is_copy_constructible_v<openmx::core::ForceHnlBatch>);
  static_assert(std::is_standard_layout_v<OpenMXForcePairItem>);
  openmx::core::OverwriteTensor<double,2> matrix({3,5});auto mapping=matrix.view();
  for(size_t i=0;i<3;++i)for(size_t j=0;j<5;++j)openmx::core::element(mapping,i,j)=10*i+j;
  require(matrix.data()[14]==24 && matrix.size()==15,"overwrite mdspan layout differs");
  auto* address=matrix.data();openmx::core::OverwriteTensor<double,2> moved(std::move(matrix));
  require(!matrix.data() && !matrix.size() && !matrix.extent(0) && moved.data()==address && moved.data()[14]==24,
      "overwrite move lost storage or left stale extents");
  matrix=std::move(moved);
  require(!moved.data() && !moved.size() && matrix.data()==address,"overwrite move assignment changed ownership");
  require(offsetof(OpenMXForcePairItem,k_off)==sizeof(size_t) &&
      offsetof(OpenMXForcePairItem,jan)==sizeof(size_t)+3*sizeof(int),"kernel pair layout changed");
  for(int kind:{1,2})for(size_t n:{size_t(0),size_t(1),size_t(513)}){
    const size_t rows=kind==1?3*n:n;
    openmx::core::ForceHnlBatch batch(kind,n,rows,n*6);
    auto v=batch.view();Probe p;
    require(v.kind==kind && v.items==n && v.rows==rows && v.density_values==6*n,"batch extents differ");
    require(operate(&v,&p),"batch fill failed");
    for(size_t i=0;i<n;++i){
      require(v.atom[i]==int(i)+1 && v.q[i]==int(i)+2,"atom metadata overlaps");
      if(kind==2)require(v.h[i]==int(i)+3,"case 2 centre metadata overlaps");
      for(int d=0;d<3;++d)require(v.force[3*i+d]==double(i)+.125*d,"force layout differs");
    }
    for(size_t i=0;i<rows;++i)
      require(v.a[i]==11*i && v.b[i]==13*i && v.row_kind[i]==int(i%3) &&
          v.energy[i]==int(i%7) && v.projectors[i]==int(i%5),"row metadata overlaps");
    require(reinterpret_cast<uintptr_t>(v.item)%alignof(OpenMXForcePairItem)==0,"pair alignment differs");
    openmx::core::ForceHnlBatch other(kind,n,rows,6*n);
    require(other.view().force!=v.force,"independent owners alias");
  }
}
void lifetime() {
  for(int kind:{1,2})for(int mode:{0,1,2,3}){
    Probe p;p.mode=mode;char error[64];const auto live=openmx::test::live_allocations();
    require(run(kind,p,error,sizeof(error))==(mode==0),"callback result changed");
    require(p.entered==1 && p.finished==1,"cleanup missing or repeated on callback exit");
    require(mode==0?!*error:bool(*error),"exception/return error was lost");
    require(openmx::test::live_allocations()==live,"host batch survived callback cleanup");
  }
  Probe p;p.mode=2;char error[1]{'x'};
  require(!run(1,p,error,sizeof(error)) && !*error,"one-byte error buffer overrun");
  p={};require(run(2,p,nullptr,0),"null error buffer rejected");
}
void invalid() {
  bool overflow=false;
  try { openmx::core::OverwriteTensor<double,2> too_large({std::numeric_limits<size_t>::max(),2}); }
  catch(const std::length_error&){overflow=true;}
  require(overflow,"overwrite tensor extent product overflow accepted");
  for(auto shape:{std::array<size_t,4>{0,1,1,1},{3,1,1,1},{1,size_t(INT_MAX)+1,0,0},
      {1,1,size_t(INT_MAX)+1,0},{2,2,1,1},{1,1,1,std::numeric_limits<size_t>::max()}}){
    Probe p;char error[256];const auto live=openmx::test::live_allocations();
    require(!openmx_core_force_hnl_batch_run(int(shape[0]),shape[1],shape[2],shape[3],operate,finish,&p,error,sizeof(error)),"invalid batch accepted");
    require(!p.entered && !p.finished && *error,"invalid shape entered native dispatch/cleanup");
    require(openmx::test::live_allocations()==live,"invalid shape leaked a partial batch");
  }
  require(!openmx_core_force_hnl_batch_run(1,1,1,1,nullptr,nullptr,nullptr,nullptr,0),"missing callback accepted");
}
void allocation() {
  for(int kind:{1,2}){
    Probe p;char error[256];openmx::test::fail_allocation_after(10000);
    const int ok=run(kind,p,error,sizeof(error));
    const auto count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    require(ok && count==6,"six-allocation layout changed");
    for(long fail=0;fail<count;++fail){
      p={};const auto live=openmx::test::live_allocations();openmx::test::fail_allocation_after(fail);
      const int accepted=run(kind,p,error,sizeof(error));openmx::test::fail_allocation_after(-1);
      require(!accepted && !p.entered && !p.finished && *error,"partial allocation entered native dispatch");
      require(openmx::test::live_allocations()==live,"partial typed batch leaked");
    }
  }
}
int nested_operation(const OpenMXForceHnlBatchView* view,void*) {
  view->force[0]=97;Probe p;p.outer=view->force;p.mode=2;char error[256];
  require(!run(view->kind,p,error,sizeof(error)) && p.finished==1,"nested failure skipped cleanup");
  require(view->force[0]==97,"nested failure altered outer storage");return 1;
}
void nested() {
  for(int kind:{1,2}){char error[256];
    require(openmx_core_force_hnl_batch_run(kind,2,2,6,nested_operation,nullptr,nullptr,error,sizeof(error)),error);
  }
}
void concurrent() {
  std::array<int,4> ok{};std::array<std::thread,4> threads;
  for(size_t t=0;t<threads.size();++t)threads[t]=std::thread([&,t]{
    for(int i=0;i<64;++i){Probe p;char error[256];if(!run(1+int(t%2),p,error,sizeof(error)) || p.finished!=1)return;}
    ok[t]=1;
  });
  for(auto& thread:threads)thread.join();for(auto value:ok)require(value==1,"concurrent batches failed");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="layouts")layouts();else if(name=="lifetime")lifetime();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="nested")nested();else if(name=="concurrent")concurrent();
    else throw std::invalid_argument("unknown nonlocal Force batch test");
    std::cout<<"PASS nonlocal Force batch "<<name<<'\n';return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
