#include <openmx_core_cwf_workspace.h>
#include "allocation_faults.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
OpenMXCWFBlocks blocks(OpenMXSolverInvocation* owner,int worker,const char* name,
    OpenMXCWFBlockShape shape,size_t bytes=sizeof(double)) {
  char error[160];OpenMXCWFBlocks out{};
  require(openmx_core_solver_cwf_blocks(owner,worker,name,&shape,bytes,&out,error,sizeof(error)) && !*error,error);
  return out;
}
using Operation=void(*)(OpenMXSolverInvocation*);
int invoke(OpenMXSolverInvocation* owner,void* opaque,double* result) {
  (*static_cast<Operation*>(opaque))(owner);*result=17;return 1;
}
void run(Operation op) {
  double result=0;char error[160];
  require(openmx_core_solver_run(4,invoke,&op,&result,error,sizeof(error)) && result==17,error);
}
void layout(OpenMXSolverInvocation* owner) {
  // Non-cubic cells, both spins, complex pairs and a padded local matrix.
  auto v=blocks(owner,0,"hopping",{2,{3,2,4},7},2*sizeof(double));
  for(size_t s=0;s<2;++s)for(size_t i=0;i<3;++i)for(size_t j=0;j<2;++j)for(size_t k=0;k<4;++k) {
    auto* cell=static_cast<unsigned char*>(openmx_cwf_block_cell(&v,s,i,j,k));
    const size_t n=((s*3+i)*2+j)*4+k;
    require(cell==static_cast<unsigned char*>(v.values)+n*7*2*sizeof(double),"cell stride changed matrix order");
    for(size_t e=0;e<7;++e) {
      const double pair[]={double(100*n+e),-double(100*n+e)};
      std::memcpy(cell+e*sizeof(pair),pair,sizeof(pair));
    }
  }
  auto* linear=static_cast<unsigned char*>(v.values);
  for(size_t n=0;n<48;++n)for(size_t e=0;e<7;++e) {
    double pair[2];std::memcpy(pair,linear+(n*7+e)*sizeof(pair),sizeof(pair));
    require(pair[0]==double(100*n+e) && pair[1]==-pair[0],"spin/cell/local-element storage overlaps");
  }
  require(reinterpret_cast<std::uintptr_t>(v.values)%alignof(std::max_align_t)==0,"C99 complex storage is misaligned");
  require(!openmx_cwf_block_cell(nullptr,0,0,0,0) && !openmx_cwf_block_cell(&v,2,0,0,0) &&
          !openmx_cwf_block_cell(&v,0,3,0,0) && !openmx_cwf_block_cell(&v,0,0,2,0) &&
          !openmx_cwf_block_cell(&v,0,0,0,4),"invalid cell index accepted");
}
void reuse(OpenMXSolverInvocation* owner) {
  auto a=blocks(owner,0,"reuse",{2,{3,2,1},5});static_cast<double*>(a.values)[59]=91.25;
  openmx::test::fail_allocation_after(0);
  auto b=blocks(owner,0,"reuse",{1,{2,3,2},5});
  require(a.values==b.values && static_cast<double*>(b.values)[59]==91.25,"same-capacity reshape lost values");
  auto c=blocks(owner,0,"reuse",{1,{1,1,1},3});
  require(c.values==a.values,"shrinking workspace allocated");
  require(openmx::test::remaining_allocations()==0,"workspace reuse allocated");
  openmx::test::fail_allocation_after(-1);
  auto other=blocks(owner,0,"other",{1,{1,1,1},3});
  require(other.values!=a.values,"different hopping buffers share storage");
}
void invalid(OpenMXSolverInvocation* owner) {
  auto live=blocks(owner,0,"valid",{1,{2,3,1},5});const auto saved=live;char error[160];
  const auto shape=live.shape;
  auto reject=[&](OpenMXSolverInvocation* s,int worker,const char* name,const OpenMXCWFBlockShape* dims,size_t bytes) {
    require(!openmx_core_solver_cwf_blocks(s,worker,name,dims,bytes,&live,error,sizeof(error)) && *error,
            "invalid block request accepted");
    require(live.values==saved.values && live.shape.spins==saved.shape.spins &&
            live.shape.elements==saved.shape.elements,"failed request changed borrowed output");
  };
  reject(nullptr,0,"valid",&shape,8);reject(owner,-1,"valid",&shape,8);reject(owner,4,"valid",&shape,8);
  reject(owner,0,nullptr,&shape,8);reject(owner,0,"",&shape,8);reject(owner,0,"valid",nullptr,8);
  reject(owner,0,"valid",&shape,0);
  require(!openmx_core_solver_cwf_blocks(owner,0,"valid",&shape,8,nullptr,error,sizeof(error)),"null output accepted");
  char tiny='x';require(!openmx_core_solver_cwf_blocks(nullptr,0,"x",&shape,8,&live,&tiny,1) && tiny=='\0',
                       "tiny diagnostic is not terminated");
  require(!openmx_core_solver_cwf_blocks(nullptr,0,"x",&shape,8,&live,nullptr,0),"diagnostic required for failure");
}
void overflow(OpenMXSolverInvocation* owner) {
  const auto max=std::numeric_limits<size_t>::max();char error[160];
  auto live=blocks(owner,0,"overflow",{1,{1,1,1},4});void* saved=live.values;
  const OpenMXCWFBlockShape shapes[]={{max,{2,1,1},1},{1,{max,2,1},1},{1,{1,1,max},2},
      {1,{1,1,1},max/sizeof(double)+1}};
  for(auto shape:shapes) {
    require(!openmx_core_solver_cwf_blocks(owner,0,"overflow",&shape,sizeof(double),&live,error,sizeof(error)) &&
            live.values==saved,"overflow replaced live hopping buffer");
  }
  // Empty extents suppress irrelevant products even if an earlier axis is huge.
  for(size_t axis=0;axis<5;++axis) {
    size_t dims[]={max,max,max,max,max};dims[axis]=0;
    auto empty=blocks(owner,0,"empty",{dims[0],{dims[1],dims[2],dims[3]},dims[4]},max);
    require(!empty.values && !openmx_cwf_block_cell(&empty,0,0,0,0),"empty view exposes storage");
  }
}
void failures(OpenMXSolverInvocation* owner) {
  auto live=blocks(owner,0,"failure",{1,{2,3,1},5});static_cast<double*>(live.values)[29]=7.75;
  bool succeeded=false;
  for(long budget=0;budget<40 && !succeeded;++budget) {
    auto out=live;char error[160];const OpenMXCWFBlockShape shape={2,{3,4,5},6};
    openmx::test::fail_allocation_after(budget);
    const int ok=openmx_core_solver_cwf_blocks(owner,0,"failure",&shape,16,&out,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if(ok)succeeded=true;
    else require(*error && out.values==live.values && static_cast<double*>(live.values)[29]==7.75,
                 "allocation failure damaged live data/output");
  }
  require(succeeded,"allocation growth never succeeded");
}
int exit_callback(OpenMXSolverInvocation* owner,void* opaque,double* result) {
  blocks(owner,0,"lifetime",{2,{3,4,5},7});
  const int action=*static_cast<int*>(opaque);
  if(action==2)throw std::runtime_error("forced callback exception");
  *result=9;return action==0;
}
void lifetime() {
  const auto initial=openmx::test::live_allocations();
  for(int action=0;action<3;++action)for(int i=0;i<40;++i) {
    double result=31;char error[160];
    const int ok=openmx_core_solver_run(1,exit_callback,&action,&result,error,sizeof(error));
    require(ok==(action==0) && result==(action==0?9:31),"callback return/exception changed result contract");
    require(openmx::test::live_allocations()==initial,"hopping allocation survived callback return");
  }
}
void concurrent(OpenMXSolverInvocation* owner) {
  std::array<int,4> ok{};std::array<std::thread,4> threads;
  for(int w=0;w<4;++w)threads[w]=std::thread([&,w] {
    try {
      auto v=blocks(owner,w,"same-name",{2,{3,2,4},5});
      auto* p=static_cast<double*>(v.values);
      for(int i=0;i<240;++i)p[i]=1000*w+i;
      for(int n=0;n<40;++n) {
        auto current=blocks(owner,w,"same-name",v.shape);
        require(current.values==p,"worker buffer moved");
        for(int i=0;i<240;++i)require(p[i]==1000*w+i,"workers share hopping values");
      }
      ok[w]=1;
    }catch(...){}
  });
  for(auto& thread:threads)thread.join();
  for(int value:ok)require(value==1,"concurrent worker failed");
}
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected one test case");const std::string_view name=argv[1];
    if(name=="layout")run(layout);else if(name=="reuse")run(reuse);else if(name=="invalid")run(invalid);
    else if(name=="overflow")run(overflow);else if(name=="failures")run(failures);
    else if(name=="lifetime")lifetime();else if(name=="concurrent")run(concurrent);
    else require(false,"unknown test case");
  }catch(const std::exception& e) {
    openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;
  }
}
